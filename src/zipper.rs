use gst::Buffer;

pub struct Zipper {
    buffers: Vec<Vec<Buffer>>,
}

impl Zipper {
    pub fn with_size(size: usize) -> Zipper {
        Zipper {
            buffers: vec![Vec::new(); size],
        }
    }

    pub fn push(&mut self, buffer: Buffer, idx: usize) {
        self.buffers[idx].push(buffer);
    }

    pub fn try_zip(&mut self) -> Option<Vec<Buffer>> {
        let filled = self.buffers.iter().filter(|&buf| buf.len() == 0).count() == 0;
        if filled {
            let mut zip = Vec::new();
            for idx in 0..self.buffers.len() {
                zip.push(self.buffers[idx].pop().unwrap());
                self.buffers[idx].clear();
            }
            Some(zip)
        } else {
            None
        }
    }
}
