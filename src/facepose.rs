use failure::Fallible;
use std::env;
use std::i32;
use std::sync::Mutex;

use crate::caps;
use crate::cata;
use crate::registry;

use glib::subclass;
use gst;
use gst_video;

use tch;
use tch::Tensor;

use crate::render;

const WIDTH: i32 = 120;
const HEIGHT: i32 = 120;

const TEXTURE_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: 256,
    height: 256,
    depth: 1,
};

lazy_static! {
    static ref IMAGENET_MEAN: Mutex<Tensor> = Mutex::new(
        Tensor::of_slice(&[127.5f32, 127.5f32, 127.5f32])
            .to_device(tch::Device::Cuda(0))
            .view((3, 1, 1))
    );
    static ref IMAGENET_STD: Mutex<Tensor> = Mutex::new(
        Tensor::of_slice(&[128f32, 128f32, 128f32])
            .to_device(tch::Device::Cuda(0))
            .view((3, 1, 1))
    );
}

pub fn normalize(tensor: &Tensor) -> Fallible<Tensor> {
    let mean = IMAGENET_MEAN.lock().unwrap();
    let std = IMAGENET_STD.lock().unwrap();
    tensor.to_kind(tch::Kind::Float).f_sub(&mean)?.f_div(&std)
}

struct Config {
    std: Tensor,
    mean: Tensor,
    u: Tensor,
    w_shp: Tensor,
    w_exp: Tensor,
    u_base: Tensor,
    w_shp_base: Tensor,
    w_exp_base: Tensor,
    tri: Vec<u32>,
}

struct Param {
    p: Tensor,
    offset: Tensor,
    alpha_shp: Tensor,
    alpha_exp: Tensor,
}

fn parse_param(param: &Tensor) -> Param {
    let pp = param.slice(0, 0, 12, 1).view((3, -1));
    let p = pp.narrow(1, 0, 3);
    let offset = pp.narrow(1, 3, 1).view((3, 1));
    let alpha_shp = param.slice(0, 12, 52, 1).view((-1, 1));
    let alpha_exp = param.slice(0, 52, 62, 1).view((-1, 1));
    Param {
        p,
        offset,
        alpha_shp,
        alpha_exp,
    }
}

lazy_static! {
    static ref CAPS_IN: Mutex<gst::Caps> = Mutex::new(gst::Caps::new_simple(
        "video/x-raw",
        &[
            (
                "format",
                &gst::List::new(&[&gst_video::VideoFormat::Bgr.to_str()]),
            ),
            ("width", &WIDTH),
            ("height", &HEIGHT),
            (
                "framerate",
                &gst::FractionRange::new(gst::Fraction::new(0, 1), gst::Fraction::new(i32::MAX, 1),),
            ),
        ],
    ));
    static ref CAPS_OUT: Mutex<gst::Caps> = Mutex::new(gst::Caps::new_simple(
        "video/x-raw",
        &[
            (
                "format",
                &gst::List::new(&[&gst_video::VideoFormat::Rgba.to_str()]),
            ),
            ("width", &256),
            ("height", &256),
            (
                "framerate",
                &gst::FractionRange::new(gst::Fraction::new(0, 1), gst::Fraction::new(i32::MAX, 1),),
            ),
        ],
    ));
    static ref FACE_MODEL: Mutex<tch::CModule> = Mutex::new(
        tch::CModule::load(env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/prediction.pt")
            .unwrap()
    );
    static ref CONFIG: Mutex<Config> = Mutex::new(Config {
        std: Tensor::read_npy(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.std.npy"
        )
        .unwrap()
        .to_device(tch::Device::Cuda(0)),
        mean: Tensor::read_npy(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.mean.npy"
        )
        .unwrap()
        .to_device(tch::Device::Cuda(0)),
        u: Tensor::read_npy(env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.u.npy")
            .unwrap()
            .to_device(tch::Device::Cuda(0)),
        w_shp: Tensor::read_npy(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.w_shp.npy"
        )
        .unwrap()
        .to_device(tch::Device::Cuda(0)),
        w_exp: Tensor::read_npy(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.w_exp.npy"
        )
        .unwrap()
        .to_device(tch::Device::Cuda(0)),
        u_base: Tensor::read_npy(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.u_base.npy"
        )
        .unwrap()
        .to_device(tch::Device::Cuda(0)),
        w_shp_base: Tensor::read_npy(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.w_shp_base.npy"
        )
        .unwrap()
        .to_device(tch::Device::Cuda(0)),
        w_exp_base: Tensor::read_npy(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.w_exp_base.npy"
        )
        .unwrap()
        .to_device(tch::Device::Cuda(0)),
        tri: tri_to_indices(env::var("SIMBOTIC_TORCH").unwrap() + "/models/facepose/param.tri.npy"),
    });
}

fn tri_to_indices(file: String) -> Vec<u32> {
    let param_tri = Tensor::read_npy(file).unwrap();
    let indices: Vec<i32> = Vec::from(param_tri);
    indices.iter().map(|i| (*i - 1) as u32).collect()
}

pub struct FacePose {
    video_info_in: gst_video::VideoInfo,
    video_info_out: gst_video::VideoInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    morph_model: render::facepose::morph::Model,
    landmarks_model: render::facepose::landmarks::Model,
    output_buffer: wgpu::Buffer,
}

impl registry::Registry for FacePose {
    const NAME: &'static str = "facepose";
    const DEBUG_CATEGORY: &'static str = "facepose";
    register_typedata!();

    fn properties() -> &'static [glib::subclass::Property<'static>] {
        &[]
    }
}

async fn gpu_setup() -> (wgpu::Device, wgpu::Queue) {
    let adapter = wgpu::Instance::new()
        .request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: None,
            },
            wgpu::BackendBit::PRIMARY,
        )
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions {
                    anisotropic_filtering: false,
                },
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    (device, queue)
}

impl std::default::Default for FacePose {
    fn default() -> Self {
        let mut caps_in: gst::Caps = CAPS_IN.lock().unwrap().clone();
        caps_in.fixate();
        let mut caps_out: gst::Caps = CAPS_OUT.lock().unwrap().clone();
        caps_out.fixate();

        let (device, queue) = futures::executor::block_on(gpu_setup());

        let morph_model =
            render::facepose::morph::model(&device, TEXTURE_EXTENT, &CONFIG.lock().unwrap().tri);

        let landmarks_model = render::facepose::landmarks::model(&device, TEXTURE_EXTENT);

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (TEXTURE_EXTENT.width * TEXTURE_EXTENT.height) as u64
                * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            label: None,
        });

        FacePose {
            video_info_in: gst_video::VideoInfo::from_caps(&caps_in).unwrap(),
            video_info_out: gst_video::VideoInfo::from_caps(&caps_out).unwrap(),
            device,
            queue,
            morph_model,
            landmarks_model,
            output_buffer,
        }
    }
}

impl caps::CapsDef for FacePose {
    fn caps_def() -> (Vec<caps::PadCaps>, Vec<caps::PadCaps>) {
        let in_caps = caps::PadCaps {
            name: "face",
            caps: CAPS_IN.lock().unwrap().clone(),
        };
        let out_morph_caps = caps::PadCaps {
            name: "morph",
            caps: CAPS_OUT.lock().unwrap().clone(),
        };
        let out_landmarks_caps = caps::PadCaps {
            name: "landmarks",
            caps: CAPS_OUT.lock().unwrap().clone(),
        };
        (vec![in_caps], vec![out_morph_caps, out_landmarks_caps])
    }
}

impl cata::Process for FacePose {
    fn process(
        &mut self,
        inbufs: &Vec<gst::Buffer>,
        outbufs: &mut Vec<gst::Buffer>,
    ) -> Result<(), std::io::Error> {
        let in_ref = inbufs[0].as_ref();
        let in_frame =
            gst_video::VideoFrameRef::from_buffer_ref_readable(in_ref, &self.video_info_in)
                .unwrap();
        let _in_stride = in_frame.plane_stride()[0] as usize;
        let _in_format = in_frame.format();
        let in_width = in_frame.width() as i32;
        let in_height = in_frame.height() as i32;
        let in_data = in_frame.plane_data(0).unwrap();

        let img_slice = unsafe {
            std::slice::from_raw_parts(in_data.as_ptr(), (in_width * in_height * 3) as usize)
        };
        let img = Tensor::of_data_size(
            img_slice,
            &[HEIGHT as i64, WIDTH as i64, 3],
            tch::Kind::Uint8,
        )
        .to_device(tch::Device::Cuda(0))
        .permute(&[2, 0, 1]);
        let img = normalize(&img).unwrap().unsqueeze(0);
        let img: tch::IValue = tch::IValue::Tensor(img);

        let face_pred = FACE_MODEL.lock().unwrap().forward_is(&[img]).unwrap();
        let face_pred = if let tch::IValue::Tensor(face_pred) = &face_pred {
            Some(face_pred)
        } else {
            None
        };
        let face_pred = face_pred.unwrap().squeeze();

        let config = CONFIG.lock().unwrap();

        let param = face_pred * &config.std + &config.mean;
        let param = parse_param(&param);

        let vertex_dense = (&config.u
            + config.w_shp.matmul(&param.alpha_shp)
            + config.w_exp.matmul(&param.alpha_exp))
        .view((3, -1));
        let stride = vertex_dense.size()[1];
        let vertex_dense = vertex_dense.as_strided(&[3, stride], &[1, 3], 0);
        let vertex_dense = param.p.matmul(&vertex_dense) + &param.offset;
        let vertex_dense = vertex_dense.as_strided(&[stride, 3], &[1, stride], 0);
        let vertices: Vec<f32> = Vec::from(vertex_dense);

        let vertex_68pts = (&config.u_base
            + config.w_shp_base.matmul(&param.alpha_shp)
            + config.w_exp_base.matmul(&param.alpha_exp))
        .view((3, -1));
        let stride = vertex_68pts.size()[1];
        let vertex_68pts = vertex_68pts.as_strided(&[3, stride], &[1, 3], 0);
        let vertex_68pts = param.p.matmul(&vertex_68pts) + &param.offset;
        let vertex_68pts = vertex_68pts.as_strided(&[stride, 3], &[1, stride], 0);
        let landmarks: Vec<f32> = Vec::from(vertex_68pts);

        let time = in_ref.get_pts();

        render::facepose::morph::update(&self.device, time, &mut self.morph_model);
        render::facepose::landmarks::update(&self.device, time, &mut self.landmarks_model);

        let out_img_size = TEXTURE_EXTENT.width;
        let out_img_size_bytes =
            (out_img_size * out_img_size) as u64 * std::mem::size_of::<u32>() as u64;

        {
            let command_buffer = {
                let mut encoder =
                    render::facepose::morph::view(&self.device, &vertices, &self.morph_model);
                encoder.copy_texture_to_buffer(
                    wgpu::TextureCopyView {
                        texture: &self.morph_model.graphics.color_texture,
                        mip_level: 0,
                        array_layer: 0,
                        origin: wgpu::Origin3d::ZERO,
                    },
                    wgpu::BufferCopyView {
                        buffer: &self.output_buffer,
                        offset: 0,
                        bytes_per_row: std::mem::size_of::<u32>() as u32 * out_img_size,
                        rows_per_image: 0,
                    },
                    TEXTURE_EXTENT,
                );
                encoder.finish()
            };

            self.queue.submit(Some(command_buffer));

            let buffer_future = self.output_buffer.map_read(0, out_img_size_bytes);

            self.device.poll(wgpu::Maintain::Wait);

            if let Ok(mapping) = futures::executor::block_on(buffer_future) {
                outbufs[0] = gst::Buffer::with_size(out_img_size_bytes as usize).unwrap();
                let out_ref = outbufs[0].get_mut().unwrap();
                out_ref.set_pts(in_ref.get_pts());
                out_ref.set_dts(in_ref.get_pts());
                out_ref.set_offset(in_ref.get_offset());
                out_ref.set_duration(in_ref.get_duration());
                let mut out_frame = gst_video::VideoFrameRef::from_buffer_ref_writable(
                    out_ref,
                    &self.video_info_out,
                )
                .unwrap();
                let out_data = out_frame.plane_data_mut(0).unwrap();
                out_data.clone_from_slice(mapping.as_slice());
            }
        }

        {
            let command_buffer = {
                let mut encoder = render::facepose::landmarks::view(
                    &self.device,
                    &landmarks,
                    &self.landmarks_model,
                );
                encoder.copy_texture_to_buffer(
                    wgpu::TextureCopyView {
                        texture: &self.landmarks_model.graphics.color_texture,
                        mip_level: 0,
                        array_layer: 0,
                        origin: wgpu::Origin3d::ZERO,
                    },
                    wgpu::BufferCopyView {
                        buffer: &self.output_buffer,
                        offset: 0,
                        bytes_per_row: std::mem::size_of::<u32>() as u32 * out_img_size,
                        rows_per_image: 0,
                    },
                    TEXTURE_EXTENT,
                );
                encoder.finish()
            };

            self.queue.submit(Some(command_buffer));

            let buffer_future = self.output_buffer.map_read(0, out_img_size_bytes);

            self.device.poll(wgpu::Maintain::Wait);

            if let Ok(mapping) = futures::executor::block_on(buffer_future) {
                outbufs[1] = gst::Buffer::with_size(out_img_size_bytes as usize).unwrap();
                let out_ref = outbufs[1].get_mut().unwrap();
                out_ref.set_pts(in_ref.get_pts());
                out_ref.set_dts(in_ref.get_pts());
                out_ref.set_offset(in_ref.get_offset());
                out_ref.set_duration(in_ref.get_duration());
                let mut out_frame = gst_video::VideoFrameRef::from_buffer_ref_writable(
                    out_ref,
                    &self.video_info_out,
                )
                .unwrap();
                let out_data = out_frame.plane_data_mut(0).unwrap();
                out_data.clone_from_slice(mapping.as_slice());
            }
        }

        Ok(())
    }

    fn set_property(&mut self, _property: &subclass::Property, _value: &glib::Value) {}
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn parse_param_() {
        let param_std = Tensor::read_npy("models/facepose/param.std.npy").unwrap();
        let param_mean = Tensor::read_npy("models/facepose/param.mean.npy").unwrap();

        let p = Tensor::read_npy("models/facepose/test.param.p.npy").unwrap();
        let offset = Tensor::read_npy("models/facepose/test.param.offset.npy").unwrap();
        let alpha_shp = Tensor::read_npy("models/facepose/test.param.alpha_shp.npy").unwrap();
        let alpha_exp = Tensor::read_npy("models/facepose/test.param.alpha_exp.npy").unwrap();

        let param = Tensor::read_npy("models/facepose/test.param.npy").unwrap();
        println!("param: {:?}", param.size());
        let param = param * param_std + param_mean;
        let param = parse_param(&param);

        assert_eq!(param.p, p);
        assert_eq!(param.offset, offset);
        assert_eq!(param.alpha_shp, alpha_shp);
        assert_eq!(param.alpha_exp, alpha_exp);
    }

    #[test]
    fn vertex_dense() {
        let param_std = Tensor::read_npy("models/facepose/param.std.npy").unwrap();
        let param_mean = Tensor::read_npy("models/facepose/param.mean.npy").unwrap();

        let u = Tensor::read_npy("models/facepose/param.u.npy").unwrap();
        let w_shp = Tensor::read_npy("models/facepose/param.w_shp.npy").unwrap();
        let w_exp = Tensor::read_npy("models/facepose/param.w_exp.npy").unwrap();

        let param = Tensor::read_npy("models/facepose/test.param.npy").unwrap();
        let param = param * param_std + param_mean;
        let param = parse_param(&param);

        let vertex_dense =
            (u + w_shp.matmul(&param.alpha_shp) + w_exp.matmul(&param.alpha_exp)).view((3, -1));
        let stride = vertex_dense.size()[1];
        let vertex_dense = vertex_dense.as_strided(&[3, stride], &[1, 3], 0);
        let vertex_dense = param.p.matmul(&vertex_dense) + param.offset;
        vertex_dense.slice(1, 0, 3, 1).print();

        let test_vertex_dense =
            Tensor::read_npy("models/facepose/test.param.vertex.dense.npy").unwrap();
        test_vertex_dense.slice(1, 0, 3, 1).print();

        assert_eq!(
            vertex_dense.slice(1, 53210, 53215, 1),
            test_vertex_dense.slice(1, 53210, 53215, 1)
        );
    }

    #[test]
    fn vertex_68pts() {
        let param_std = Tensor::read_npy("models/facepose/param.std.npy").unwrap();
        let param_mean = Tensor::read_npy("models/facepose/param.mean.npy").unwrap();

        let u_base = Tensor::read_npy("models/facepose/param.u_base.npy").unwrap();
        let w_shp_base = Tensor::read_npy("models/facepose/param.w_shp_base.npy").unwrap();
        let w_exp_base = Tensor::read_npy("models/facepose/param.w_exp_base.npy").unwrap();

        let param = Tensor::read_npy("models/facepose/test.param.npy").unwrap();
        let param = param * param_std + param_mean;
        let param = parse_param(&param);

        let vertex_68pts =
            (u_base + w_shp_base.matmul(&param.alpha_shp) + w_exp_base.matmul(&param.alpha_exp))
                .view((3, -1));
        let stride = vertex_68pts.size()[1];
        let vertex_68pts = vertex_68pts.as_strided(&[3, stride], &[1, 3], 0);
        let vertex_68pts = param.p.matmul(&vertex_68pts) + param.offset;

        let test_vertex_68pts =
            Tensor::read_npy("models/facepose/test.param.vertex.68pts.npy").unwrap();

        assert_eq!(
            vertex_68pts.slice(1, 0, 5, 1),
            test_vertex_68pts.slice(1, 0, 5, 1)
        );
    }
}
