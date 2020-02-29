use crate::caps;
use crate::cata;
use crate::registry;
use failure::Fallible;
use gst;
use gst_video;
use std::i32;
use std::sync::Mutex;
use tch;
use tch::Tensor;

fn lerp<T: num_traits::float::Float>(a: T, b: T, alpha: T) -> T {
    a + alpha * (b - a)
}

fn tensor_map_range(
    val: &Tensor,
    in_a: &Tensor,
    in_b: &Tensor,
    out_a: &Tensor,
    out_b: &Tensor,
) -> Fallible<Tensor> {
    let pos = val.f_sub(in_a)?.f_div(&in_b.f_sub(in_a)?)?;
    let mapped = out_a.f_add(&out_b.f_sub(out_a)?.f_mul(&pos)?)?;
    Ok(mapped.clamp(f64::from(out_a), f64::from(out_b)))
}

const WIDTH: i32 = 640;
const HEIGHT: i32 = 192;

lazy_static! {
    static ref CAPS: Mutex<gst::Caps> = Mutex::new(gst::Caps::new_simple(
        "video/x-raw",
        &[
            (
                "format",
                &gst::List::new(&[&gst_video::VideoFormat::Rgb.to_str()]),
            ),
            ("width", &WIDTH),
            ("height", &HEIGHT),
            (
                "framerate",
                &gst::FractionRange::new(gst::Fraction::new(0, 1), gst::Fraction::new(i32::MAX, 1),),
            ),
        ],
    ));
    static ref ENCODER_MODEL: Mutex<tch::CModule> =
        Mutex::new(tch::CModule::load("models/monodepth/encoder.pt").unwrap());
    static ref DECODER_MODEL: Mutex<tch::CModule> =
        Mutex::new(tch::CModule::load("models/monodepth/decoder.pt").unwrap());
}

pub struct MonoDepth {
    video_info: gst_video::VideoInfo,
    color_map: Tensor, // Tensor[[3, 1, 728], Uint8]
    depth_min: f32,
    depth_max: f32,
}

impl registry::Registry for MonoDepth {
    const NAME: &'static str = "monodepth";
    const DEBUG_CATEGORY: &'static str = "monodepth";
    register_typedata!();
}

impl std::default::Default for MonoDepth {
    fn default() -> Self {
        let caps = gst::Caps::fixate(CAPS.lock().unwrap().clone());
        MonoDepth {
            video_info: gst_video::VideoInfo::from_caps(&caps).unwrap(),
            color_map: tch::vision::image::load("assets/magma.png")
                .unwrap()
                .to_device(tch::Device::Cuda(0)),
            depth_min: 0f32,
            depth_max: 1f32,
        }
    }
}

impl caps::CapsDef for MonoDepth {
    fn caps_def() -> (Vec<caps::PadCaps>, Vec<caps::PadCaps>) {
        let in_caps = caps::PadCaps {
            name: "rgb",
            caps: CAPS.lock().unwrap().clone(),
        };
        let out_caps = caps::PadCaps {
            name: "depth",
            caps: CAPS.lock().unwrap().clone(),
        };
        (vec![in_caps], vec![out_caps])
    }
}

impl cata::Process for MonoDepth {
    fn process(
        &mut self,
        inbuf: &Vec<gst::Buffer>,
        outbuf: &mut Vec<gst::Buffer>,
    ) -> Result<(), std::io::Error> {
        for (i, buf) in inbuf.iter().enumerate() {
            if i < outbuf.len() {
                outbuf[i] = buf.clone();
            }
        }

        let mut depth_buf = inbuf[0].copy();
        {
            let rgb_ref = inbuf[0].as_ref();
            let in_frame =
                gst_video::VideoFrameRef::from_buffer_ref_readable(rgb_ref, &self.video_info)
                    .unwrap();
            let _in_stride = in_frame.plane_stride()[0] as usize;
            let _in_format = in_frame.format();
            let _in_width = in_frame.width() as i32;
            let _in_height = in_frame.height() as i32;
            let in_data = in_frame.plane_data(0).unwrap();

            let depth_ref = depth_buf.get_mut().unwrap();
            let mut out_frame =
                gst_video::VideoFrameRef::from_buffer_ref_writable(depth_ref, &self.video_info)
                    .unwrap();
            let _out_stride = out_frame.plane_stride()[0] as usize;
            let _out_format = out_frame.format();
            let out_data = out_frame.plane_data_mut(0).unwrap();

            let img_slice = unsafe {
                std::slice::from_raw_parts(in_data.as_ptr(), (WIDTH * HEIGHT * 3) as usize)
            };
            let img = Tensor::of_data_size(
                img_slice,
                &[HEIGHT as i64, WIDTH as i64, 3],
                tch::Kind::Uint8,
            )
            .to_device(tch::Device::Cuda(0))
            .permute(&[2, 0, 1])
            .to_kind(tch::Kind::Float)
                / 255;
            let i_img: tch::IValue = tch::IValue::Tensor(img.unsqueeze(0));
            let encoder_output = ENCODER_MODEL.lock().unwrap().forward_is(&[i_img]).unwrap();
            let enc_tensors = match &encoder_output {
                tch::IValue::Tuple(enc_tensors) => Some(enc_tensors),
                _ => None,
            }
            .unwrap();

            let depth_outputs = DECODER_MODEL
                .lock()
                .unwrap()
                .forward_is(&[
                    &enc_tensors[0],
                    &enc_tensors[1],
                    &enc_tensors[2],
                    &enc_tensors[3],
                    &enc_tensors[4],
                ])
                .unwrap();

            // Tensor[[1, 1, 192, 640], Float]
            let mut depth_output = None;
            if let tch::IValue::Tuple(tensors) = &depth_outputs {
                if let tch::IValue::Tensor(tensor) = &tensors[0] {
                    depth_output = Some(tensor);
                }
            };

            let depth_output = depth_output.unwrap();
            let depth_min = depth_output.min();
            let depth_max = depth_output.max();
            self.depth_min = lerp(self.depth_min, f32::from(depth_min), 0.1f32);
            self.depth_max = lerp(self.depth_max, f32::from(depth_max), 0.1f32);

            let depth_min = Tensor::from(self.depth_min).to_device(tch::Device::Cuda(0));
            let depth_max = Tensor::from(self.depth_max).to_device(tch::Device::Cuda(0));
            let depth_map_min = Tensor::from(0f64).to_device(tch::Device::Cuda(0));
            let depth_map_max = Tensor::from(1f64).to_device(tch::Device::Cuda(0));
            let depth_output = tensor_map_range(
                depth_output,
                &depth_min,
                &depth_max,
                &depth_map_min,
                &depth_map_max,
            )
            .unwrap();

            let color_index = depth_output
                .f_mul(&Tensor::from(727f32))
                .unwrap()
                .flatten(0, 3)
                .to_kind(tch::Kind::Int64);

            let depth_color = self
                .color_map
                .index_select(2, &color_index)
                .permute(&[2, 1, 0])
                .to_device(tch::Device::Cpu);

            let depth_out = unsafe {
                std::slice::from_raw_parts_mut(out_data.as_mut_ptr(), (WIDTH * HEIGHT * 3) as usize)
            };
            depth_color
                .to_kind(tch::Kind::Uint8)
                .copy_data(depth_out, (WIDTH * HEIGHT * 3) as usize);
        }

        outbuf[0] = depth_buf;

        Ok(())
    }
}
