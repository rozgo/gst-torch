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

const WIDTH: i32 = 640;
const HEIGHT: i32 = 192;

lazy_static! {
    static ref IMAGENET_MEAN: Mutex<Tensor> =
        Mutex::new(Tensor::of_slice(&[0.485f32, 0.456, 0.406])
        .to_device(tch::Device::Cuda(0))
        .view((3, 1, 1)));
    static ref IMAGENET_STD: Mutex<Tensor> =
        Mutex::new(Tensor::of_slice(&[0.229f32, 0.224, 0.225])
        .to_device(tch::Device::Cuda(0))
        .view((3, 1, 1)));
}

pub fn normalize(tensor: &Tensor) -> Fallible<Tensor> {
    let mean = IMAGENET_MEAN.lock().unwrap();
    let std = IMAGENET_STD.lock().unwrap();
    (tensor.to_kind(tch::Kind::Float) / 255.0)
        .f_sub(&mean)?
        .f_div(&std)
}

fn label_map() -> Tensor {
    let mut labels = vec![vec![30, 15, 60]; 19];
    labels[ 0] = vec![128,  64, 128];  // 'road'
    labels[ 1] = vec![244,  35, 232];  // 'sidewalk'
    labels[ 2] = vec![ 70,  70,  70];  // 'building'
    labels[ 3] = vec![102, 102, 156];  // 'wall'
    labels[ 4] = vec![190, 153, 153];  // 'fence'
    labels[ 5] = vec![153, 153, 153];  // 'pole'
    labels[ 6] = vec![250, 170,  30];  // 'traffic light'
    labels[ 7] = vec![220, 220,   0];  // 'traffic sign'
    labels[ 8] = vec![107, 142,  35];  // 'vegetation'
    labels[ 9] = vec![152, 251, 152];  // 'terrain'
    labels[10] = vec![ 70, 130, 180];  // 'sky'
    labels[11] = vec![220,  20,  60];  // 'person'
    labels[12] = vec![255,   0,   0];  // 'rider'
    labels[13] = vec![  0,   0, 142];  // 'car'
    labels[14] = vec![  0,   0,  70];  // 'truck'
    labels[15] = vec![  0,  60, 100];  // 'bus'
    labels[16] = vec![  0,  80, 100];  // 'train'
    labels[17] = vec![  0,   0, 230];  // 'motorcycle'
    labels[18] = vec![119,  11,  32];  // 'bicycle'
    
    let labels = labels.into_iter().flatten().collect::<Vec<u8>>();
    Tensor::of_slice(&labels)
        .reshape(&[19, 1, 3])
        .permute(&[2, 1, 0])
}

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
    static ref SEMSEG_MODEL: Mutex<tch::CModule> = Mutex::new(
        tch::CModule::load(env::var("SIMBOTIC_TORCH").unwrap() + "/models/semseg/semseg.pt")
            .unwrap()
    );
}

pub struct SemSeg {
    video_info: gst_video::VideoInfo,
    color_map: Tensor, // Tensor[[3, 1, 728], Uint8]
}

impl registry::Registry for SemSeg {
    const NAME: &'static str = "semseg";
    const DEBUG_CATEGORY: &'static str = "semseg";
    register_typedata!();

    fn properties() -> &'static [glib::subclass::Property<'static>] {
        &[]
    }
}

impl std::default::Default for SemSeg {
    fn default() -> Self {
        let mut caps: gst::Caps = CAPS.lock().unwrap().clone();
        caps.fixate();
        SemSeg {
            video_info: gst_video::VideoInfo::from_caps(&caps).unwrap(),
            color_map: label_map().to_device(tch::Device::Cuda(0)),
        }
    }
}

impl caps::CapsDef for SemSeg {
    fn caps_def() -> (Vec<caps::PadCaps>, Vec<caps::PadCaps>) {
        let in_caps = caps::PadCaps {
            name: "rgb",
            caps: CAPS.lock().unwrap().clone(),
        };
        let out_caps = caps::PadCaps {
            name: "semseg",
            caps: CAPS.lock().unwrap().clone(),
        };
        (vec![in_caps], vec![out_caps])
    }
}

impl cata::Process for SemSeg {
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

        let mut semseg_buf = inbuf[0].copy();
        {
            let rgb_ref = inbuf[0].as_ref();
            let in_frame =
                gst_video::VideoFrameRef::from_buffer_ref_readable(rgb_ref, &self.video_info)
                    .unwrap();
            let _in_stride = in_frame.plane_stride()[0] as usize;
            let _in_format = in_frame.format();
            let in_width = in_frame.width() as i32;
            let in_height = in_frame.height() as i32;
            let in_data = in_frame.plane_data(0).unwrap();

            let semseg_ref = semseg_buf.get_mut().unwrap();
            let mut out_frame =
                gst_video::VideoFrameRef::from_buffer_ref_writable(semseg_ref, &self.video_info)
                    .unwrap();
            let _out_stride = out_frame.plane_stride()[0] as usize;
            let _out_format = out_frame.format();
            let out_data = out_frame.plane_data_mut(0).unwrap();

            let img_slice = unsafe {
                std::slice::from_raw_parts(in_data.as_ptr(), in_data.len())
            };
            let img = Tensor::of_data_size(
                img_slice,
                &[in_height as i64, in_width as i64, 3],
                tch::Kind::Uint8,
            )
            .to_device(tch::Device::Cuda(0))
            .permute(&[2, 0, 1]);
            let img = normalize(&img).unwrap();
            let img: tch::IValue = tch::IValue::Tensor(img.unsqueeze(0));

            let semseg_pred = SEMSEG_MODEL.lock().unwrap().forward_is(&[img]).unwrap();
            let semseg_pred = if let tch::IValue::Tensor(semseg_pred) = &semseg_pred {
                Some(semseg_pred)
            } else {
                None
            };
            let semseg_pred = semseg_pred.unwrap().squeeze();
            let semseg_pred = semseg_pred.argmax(0, false).to_kind(tch::Kind::Uint8);

            let color_index = semseg_pred.flatten(0, 1).to_kind(tch::Kind::Int64);

            let semseg_color = self
                .color_map
                .index_select(2, &color_index)
                .permute(&[2, 1, 0])
                .to_device(tch::Device::Cpu);

            let semseg_out = unsafe {
                std::slice::from_raw_parts_mut(out_data.as_mut_ptr(), (WIDTH * HEIGHT * 3) as usize)
            };
            semseg_color
                .to_kind(tch::Kind::Uint8)
                .copy_data(semseg_out, (WIDTH * HEIGHT * 3) as usize);
        }

        outbuf[0] = semseg_buf;

        Ok(())
    }

    fn set_property(&mut self, _property: &subclass::Property, _value: &glib::Value) {}
}
