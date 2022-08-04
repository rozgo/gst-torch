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
use tch::{TchError, Tensor};

lazy_static! {
    static ref IMAGENET_MEAN: Mutex<Tensor> = Mutex::new(
        Tensor::of_slice(&[0.485f32, 0.456, 0.406])
            .to_device(tch::Device::Cuda(0))
            .view((3, 1, 1))
    );
    static ref IMAGENET_STD: Mutex<Tensor> = Mutex::new(
        Tensor::of_slice(&[0.229f32, 0.224, 0.225])
            .to_device(tch::Device::Cuda(0))
            .view((3, 1, 1))
    );
}

pub fn normalize(tensor: &Tensor) -> Result<Tensor, TchError> {
    let mean = IMAGENET_MEAN.lock().unwrap();
    let std = IMAGENET_STD.lock().unwrap();
    (tensor.to_kind(tch::Kind::Float) / 255.0)
        .f_sub(&mean)?
        .f_div(&std)
}

const WIDTH: i32 = 320;
const HEIGHT: i32 = 320;

lazy_static! {
    static ref CAPS_IN: Mutex<gst::Caps> = Mutex::new(gst::Caps::new_simple(
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
    static ref CAPS_OUT: Mutex<gst::Caps> = Mutex::new(gst::Caps::new_simple(
        "video/x-raw",
        &[
            (
                "format",
                &gst::List::new(&[&gst_video::VideoFormat::Gray8.to_str()]),
            ),
            ("width", &WIDTH),
            ("height", &HEIGHT),
            (
                "framerate",
                &gst::FractionRange::new(gst::Fraction::new(0, 1), gst::Fraction::new(i32::MAX, 1),),
            ),
        ],
    ));
    static ref MODEL: Mutex<tch::CModule> = Mutex::new(
        tch::CModule::load(env::var("SIMBOTIC_TORCH").unwrap() + "/models/salientobject/model.pt")
            .unwrap()
    );
}

pub struct SalientObject {
    video_info_in: gst_video::VideoInfo,
    video_info_out: gst_video::VideoInfo,
}

impl registry::Registry for SalientObject {
    const NAME: &'static str = "salientobject";
    const DEBUG_CATEGORY: &'static str = "salientobject";
    register_typedata!();

    fn properties() -> &'static [glib::subclass::Property<'static>] {
        &[]
    }
}

impl std::default::Default for SalientObject {
    fn default() -> Self {
        let mut caps_in: gst::Caps = CAPS_IN.lock().unwrap().clone();
        caps_in.fixate();
        let mut caps_out: gst::Caps = CAPS_OUT.lock().unwrap().clone();
        caps_out.fixate();
        SalientObject {
            video_info_in: gst_video::VideoInfo::from_caps(&caps_in).unwrap(),
            video_info_out: gst_video::VideoInfo::from_caps(&caps_out).unwrap(),
        }
    }
}

impl caps::CapsDef for SalientObject {
    fn caps_def() -> (Vec<caps::PadCaps>, Vec<caps::PadCaps>) {
        let in_caps = caps::PadCaps {
            name: "rgb",
            caps: CAPS_IN.lock().unwrap().clone(),
        };
        let out_caps = caps::PadCaps {
            name: "mask",
            caps: CAPS_OUT.lock().unwrap().clone(),
        };
        (vec![in_caps], vec![out_caps])
    }
}

impl cata::Process for SalientObject {
    fn process(
        &mut self,
        inbuf: &Vec<gst::Buffer>,
        outbufs: &mut Vec<gst::Buffer>,
    ) -> Result<(), std::io::Error> {
        let in_ref = inbuf[0].as_ref();
        let in_frame =
            gst_video::VideoFrameRef::from_buffer_ref_readable(in_ref, &self.video_info_in)
                .unwrap();
        let _in_stride = in_frame.plane_stride()[0] as usize;
        let _in_format = in_frame.format();
        let in_width = in_frame.width() as i32;
        let in_height = in_frame.height() as i32;
        let in_data = in_frame.plane_data(0).unwrap();

        let img_slice = unsafe { std::slice::from_raw_parts(in_data.as_ptr(), in_data.len()) };
        let img = Tensor::of_data_size(
            img_slice,
            &[in_height as i64, in_width as i64, 3],
            tch::Kind::Uint8,
        )
        .to_device(tch::Device::Cuda(0))
        .permute(&[2, 0, 1]);
        let img = normalize(&img).unwrap();
        let img = img.unsqueeze(0);

        let i_img: tch::IValue = tch::IValue::Tensor(img);
        let model_output = MODEL.lock().unwrap().forward_is(&[i_img]).unwrap();
        let model_tensors = match &model_output {
            tch::IValue::Tuple(enc_tensors) => Some(enc_tensors),
            _ => None,
        }
        .unwrap();

        let prediction: &tch::IValue = &model_tensors[0];
        let prediction = match &prediction {
            tch::IValue::Tensor(tensor) => Some(tensor),
            _ => None,
        }
        .unwrap();

        let prediction = prediction.squeeze().unsqueeze(0);
        let prediction: Tensor = {
            let max = tch::Tensor::max(&prediction);
            let min = tch::Tensor::min(&prediction);
            (&prediction - &min) / (&max - &min)
        };
        let prediction = prediction * Tensor::of_slice(&[255f32]).to_device(tch::Device::Cuda(0));

        outbufs[0] = gst::Buffer::with_size((WIDTH * HEIGHT) as usize).unwrap();
        let out_ref = outbufs[0].get_mut().unwrap();
        out_ref.set_pts(in_ref.get_pts());
        out_ref.set_dts(in_ref.get_pts());
        out_ref.set_offset(in_ref.get_offset());
        out_ref.set_duration(in_ref.get_duration());
        let mut out_frame =
            gst_video::VideoFrameRef::from_buffer_ref_writable(out_ref, &self.video_info_out)
                .unwrap();
        let out_data = out_frame.plane_data_mut(0).unwrap();
        let pred_out = unsafe {
            std::slice::from_raw_parts_mut(out_data.as_mut_ptr(), (WIDTH * HEIGHT) as usize)
        };
        prediction
            .to_kind(tch::Kind::Uint8)
            .copy_data(pred_out, (WIDTH * HEIGHT) as usize);

        Ok(())
    }

    fn set_property(&mut self, _property: &subclass::Property, _value: &glib::Value) {}
}
