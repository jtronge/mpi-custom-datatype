use crate::buffer::ComplexVec;

struct Random {
    last: usize,
}

impl Random {
    fn new(seed: usize) -> Random {
        Random {
            last: seed,
        }
    }

    fn value(&mut self) -> usize {
        let v = (503 * self.last) % 54018521;
        self.last = v;
        v
    }
}

/// Generate a randomly partitioned complex vec type.
pub fn generate_complex_vec(total: usize, seed: usize) -> ComplexVec {
    let mut rand = Random::new(seed);
    let mut count = 0;
    let mut data = vec![];
    while count < total {
        let len = rand.value() % total;
        let len = if (count + len) > total {
            total - count
        } else {
            len
        };
        let inner_data = (0..len).map(|i| (count + i) as i32).collect();
        data.push(inner_data);
        count += len;
    }
    ComplexVec(data)
}
