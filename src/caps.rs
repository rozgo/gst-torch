
#[derive(Debug)]
pub struct PadCaps {
    pub name: &'static str,
    pub caps: gst::Caps,
}

pub trait CapsDef {
    fn caps_def() -> (Vec<PadCaps>, Vec<PadCaps>);
}
