pub struct Random {
    last: usize,
}

impl Random {
    pub fn new(seed: usize) -> Random {
        Random {
            last: seed,
        }
    }

    /// Implements a very simple random algorithm suitable for these
    /// benchmarks.
    pub fn value(&mut self) -> usize {
        let v = (503 * self.last) % 54018521;
        self.last = v;
        v
    }
}
