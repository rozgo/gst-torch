extern crate gstreamer as gst;
use gst::prelude::*;
use gst::ElementExt;

use clap::{App, AppSettings, Arg};

#[path = "runner.rs"]
mod runner;

fn launch_main(pipeline_str: String) {
    gst::init().unwrap();

    let mut context = gst::ParseContext::new();
    let pipeline =
        match gst::parse_launch_full(&pipeline_str, Some(&mut context), gst::ParseFlags::NONE) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                if let Some(gst::ParseError::NoSuchElement) = err.kind::<gst::ParseError>() {
                    println!("Missing element(s): {:?}", context.get_missing_elements());
                } else {
                    println!("Failed to parse pipeline: {}", err);
                }
                return;
            }
        };
    let pipeline = pipeline.dynamic_cast::<gst::Pipeline>().unwrap();
    let bus = pipeline.get_bus().unwrap();

    pipeline
        .set_state(gst::State::Playing)
        .expect("Unable to set the pipeline to the `Playing` state");

    // let mut _last_face = gst::ClockTime(Some(0));

    for msg in bus.iter_timed(gst::CLOCK_TIME_NONE) {
        use gst::MessageView;

        // let stc = msg.get_structure();
        // if let Some(stc) = stc {
        //     let name = stc.get_name();
        //     if name == "facedetect" {
        //         let faces = stc.get_value("faces");
        //         if let Ok(faces) = faces {
        //             let faces = faces.get::<gst::List>();
        //             if let Ok(Some(faces)) = faces {
        //                 let faces = faces.as_slice().len();
        //                 if faces > 0 {
        //                     let timestamp = stc
        //                         .get_value("timestamp")
        //                         .unwrap()
        //                         .get::<gst::ClockTime>()
        //                         .unwrap()
        //                         .unwrap();
        //                     _last_face = timestamp;
        //                     println!("timestamp: {:?}", timestamp);
        //                 }
        //             }
        //         }
        //     }
        // }

        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                println!(
                    "Error from {:?}: {} ({:?})",
                    err.get_src().map(|s| s.get_path_string()),
                    err.get_error(),
                    err.get_debug()
                );
                break;
            }
            _ => (),
        }
    }

    pipeline
        .set_state(gst::State::Null)
        .expect("Unable to set the pipeline to the `Null` state");
}

fn main() {

    let options = App::new("Simbotic")
        .setting(AppSettings::TrailingVarArg)
        .arg(Arg::with_name("pipeline").multiple(true))
        .get_matches();

    let pipeline: Vec<&str> = options.values_of("pipeline").unwrap().collect();
    let pipeline = pipeline.join(" ");

    let launch_handle = std::thread::spawn(move || {
        runner::run(|| launch_main(pipeline));
    });

    launch_handle.join().unwrap();
}
