use crate::caps;
use crate::cata;
use crate::registry;

use gst;

pub struct Identity {}

impl registry::Registry for Identity {
    const NAME: &'static str = "tchidentity";
    const DEBUG_CATEGORY: &'static str = "tchidentity";
    register_typedata!();
}

impl std::default::Default for Identity {
    fn default() -> Self {
        Identity {}
    }
}

impl caps::CapsDef for Identity {
    fn caps_def() -> (Vec<caps::PadCaps>, Vec<caps::PadCaps>) {
        let in_caps = caps::PadCaps {
            name: "in_any",
            caps: gst::Caps::fixate(gst::Caps::new_any()),
        };
        let out_caps = caps::PadCaps {
            name: "out_any",
            caps: gst::Caps::fixate(gst::Caps::new_any()),
        };
        (vec![in_caps], vec![out_caps])
    }
}

impl cata::Process for Identity {
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
        Ok(())
    }
}
