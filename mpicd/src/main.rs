//! Dummy example working with the Rust mpicd code.
use mpicd::communicator::Communicator;

fn main() {
    let ctx = mpicd::init().expect("failed to init mpirs");
    let size = ctx.size();
    let rank = ctx.rank();

    assert_eq!(size, 2);

    if rank == 0 {
        unsafe {
            let mut data = vec![vec![0; 128]; 128];
            let mut reqs = vec![];
            for row in &mut data {
                reqs.push(ctx.irecv(&mut row[..], 1, 0).unwrap());
            }
            let _ = ctx.waitall(&reqs);
            println!("data[0] = {:?}", &data[0][..10]);
        }
    } else {
        unsafe {
            let data: Vec<u32> = (0..128).collect();
            let mut reqs = vec![];
            for _ in 0..128 {
                reqs.push(ctx.isend(&data[..], 0, 0).unwrap());
            }
            let _ = ctx.waitall(&reqs);
        }
    }
}
