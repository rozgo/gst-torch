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

const WIDTH: i32 = 256;
const HEIGHT: i32 = 256;

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
    static ref DETECTOR_MODEL: Mutex<tch::CModule> = Mutex::new(
        tch::CModule::load(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/motiontransfer/detector.pt"
        )
        .unwrap()
    );
    static ref GENERATOR_MODEL: Mutex<tch::CModule> = Mutex::new(
        tch::CModule::load(
            env::var("SIMBOTIC_TORCH").unwrap() + "/models/motiontransfer/generator.pt"
        )
        .unwrap()
    );
}

// Metadata for the properties
static PROPERTIES: [subclass::Property; 1] = [subclass::Property("source-image", |name| {
    glib::ParamSpec::string(
        name,
        "Source image",
        "Source image to be driven",
        None,
        glib::ParamFlags::READWRITE,
    )
})];

pub struct MotionTransfer {
    video_info: gst_video::VideoInfo,
    source_image: Option<Tensor>, // Tensor[[3, 256, 256], Uint8]
    source: Option<Tensor>,       // Tensor[[1, 3, 256, 256], Float]
    kp_source: Option<(Tensor, Tensor)>,
    kp_driving_initial: Option<(Tensor, Tensor)>,
}

impl registry::Registry for MotionTransfer {
    const NAME: &'static str = "motiontransfer";
    const DEBUG_CATEGORY: &'static str = "motiontransfer";
    register_typedata!();

    fn properties() -> &'static [glib::subclass::Property<'static>] {
        &PROPERTIES
    }
}

impl std::default::Default for MotionTransfer {
    fn default() -> Self {
        let mut caps: gst::Caps = CAPS.lock().unwrap().clone();
        caps.fixate();
        MotionTransfer {
            video_info: gst_video::VideoInfo::from_caps(&caps).unwrap(),
            source_image: None,
            source: None,
            kp_source: None,
            kp_driving_initial: None,
        }
    }
}

impl caps::CapsDef for MotionTransfer {
    fn caps_def() -> (Vec<caps::PadCaps>, Vec<caps::PadCaps>) {
        let in_caps = caps::PadCaps {
            name: "rgb",
            caps: CAPS.lock().unwrap().clone(),
        };
        let out_caps = caps::PadCaps {
            name: "driven",
            caps: CAPS.lock().unwrap().clone(),
        };
        (vec![in_caps], vec![out_caps])
    }
}

impl cata::Process for MotionTransfer {
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

        let mut driven_buf = inbuf[0].copy();
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

            let driven_ref = driven_buf.get_mut().unwrap();
            let mut out_frame =
                gst_video::VideoFrameRef::from_buffer_ref_writable(driven_ref, &self.video_info)
                    .unwrap();
            let _out_stride = out_frame.plane_stride()[0] as usize;
            let _out_format = out_frame.format();
            let out_data = out_frame.plane_data_mut(0).unwrap();

            let img_slice = unsafe {
                std::slice::from_raw_parts(in_data.as_ptr(), (WIDTH * HEIGHT * 3) as usize)
            };
            let img_bytes = Tensor::of_data_size(
                img_slice,
                &[HEIGHT as i64, WIDTH as i64, 3],
                tch::Kind::Uint8,
            )
            .to_device(tch::Device::Cuda(0))
            .permute(&[2, 0, 1]);

            let driving_frame = img_bytes.to_kind(tch::Kind::Float) / 255;

            if let Some(source_image) = &self.source_image {
                if self.kp_source == None {
                    let source = source_image.to_kind(tch::Kind::Float) / 255;
                    let source = source.unsqueeze(0);
                    self.source = Some(source.copy());
                    let source = tch::IValue::Tensor(source);
                    let detector_output = DETECTOR_MODEL
                        .lock()
                        .unwrap()
                        .forward_is(&[source])
                        .unwrap();
                    let det_tensors = match &detector_output {
                        tch::IValue::Tuple(det_tensors) => Some(det_tensors),
                        _ => None,
                    }
                    .unwrap();
                    match (&det_tensors[0], &det_tensors[1]) {
                        (tch::IValue::Tensor(value), tch::IValue::Tensor(jacobian)) => {
                            self.kp_source = Some((value.copy(), jacobian.copy()))
                        }
                        _ => (),
                    };
                }
            }

            if self.kp_driving_initial == None {
                let driving_initial = tch::IValue::Tensor(driving_frame.unsqueeze(0));
                let detector_output = DETECTOR_MODEL
                    .lock()
                    .unwrap()
                    .forward_is(&[driving_initial])
                    .unwrap();
                let det_tensors = match &detector_output {
                    tch::IValue::Tuple(det_tensors) => Some(det_tensors),
                    _ => None,
                }
                .unwrap();
                match (&det_tensors[0], &det_tensors[1]) {
                    (tch::IValue::Tensor(value), tch::IValue::Tensor(jacobian)) => {
                        self.kp_driving_initial = Some((value.copy(), jacobian.copy()))
                    }
                    _ => (),
                };
            };

            let mut kp_driving: Option<(Tensor, Tensor)> = None;
            {
                let kp_driving_img = tch::IValue::Tensor(driving_frame.unsqueeze(0));
                let detector_output = DETECTOR_MODEL
                    .lock()
                    .unwrap()
                    .forward_is(&[kp_driving_img])
                    .unwrap();
                let det_tensors = match &detector_output {
                    tch::IValue::Tuple(det_tensors) => Some(det_tensors),
                    _ => None,
                }
                .unwrap();
                match (&det_tensors[0], &det_tensors[1]) {
                    (tch::IValue::Tensor(value), tch::IValue::Tensor(jacobian)) => {
                        kp_driving = Some((value.copy(), jacobian.copy()))
                    }
                    _ => (),
                };
            }

            let mut prediction: Option<Tensor> = None;
            match (
                &self.source,
                &self.kp_source,
                &kp_driving,
                &self.kp_driving_initial,
            ) {
                (
                    Some(source),
                    Some((kp_source_value, kp_source_jacobian)),
                    Some((kp_driving_value, kp_driving_jacobian)),
                    Some((kp_driving_initial_value, kp_driving_initial_jacobian)),
                ) => {
                    let kp_value_diff = kp_driving_value - kp_driving_initial_value;
                    let kp_driving_value = kp_value_diff + kp_source_value;

                    let kp_driving_initial_jacobian_inv = kp_driving_initial_jacobian.inverse();
                    let jacobian_diff =
                        kp_driving_jacobian.matmul(&kp_driving_initial_jacobian_inv);
                    let kp_driving_jacobian = jacobian_diff.matmul(&kp_source_jacobian);

                    let source = tch::IValue::Tensor(source.copy());
                    let kp_source_value = tch::IValue::Tensor(kp_source_value.copy());
                    let kp_source_jacobian = tch::IValue::Tensor(kp_source_jacobian.copy());
                    let kp_driving_value = tch::IValue::Tensor(kp_driving_value);
                    let kp_driving_jacobian = tch::IValue::Tensor(kp_driving_jacobian);

                    let gen_pred = GENERATOR_MODEL
                        .lock()
                        .unwrap()
                        .forward_is(&[
                            source,
                            kp_source_value,
                            kp_source_jacobian,
                            kp_driving_value,
                            kp_driving_jacobian,
                        ])
                        .unwrap();
                    let gen_pred = if let tch::IValue::Tensor(gen_pred) = &gen_pred {
                        Some(gen_pred)
                    } else {
                        None
                    };
                    prediction = Some(gen_pred.unwrap().squeeze());
                }
                _ => (),
            };

            let driven_out = unsafe {
                std::slice::from_raw_parts_mut(out_data.as_mut_ptr(), (WIDTH * HEIGHT * 3) as usize)
            };

            if let Some(prediction) = prediction {
                let prediction = prediction * 255;
                prediction
                    .to_kind(tch::Kind::Uint8)
                    .permute(&[1, 2, 0])
                    .copy_data(driven_out, (WIDTH * HEIGHT * 3) as usize);
            }
        }

        outbuf[0] = driven_buf;

        Ok(())
    }

    fn set_property(&mut self, property: &subclass::Property, value: &glib::Value) {
        match property {
            subclass::Property("source-image", ..) => {
                let source_path: String = value.get().expect("source image path").unwrap();
                self.source_image = Some(
                    tch::vision::image::load(source_path)
                        .unwrap()
                        .to_device(tch::Device::Cuda(0)),
                );
                self.kp_source = None;
            }
            _ => unimplemented!(),
        }
    }
}
