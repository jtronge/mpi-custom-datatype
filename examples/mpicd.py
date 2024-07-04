import atexit
import cffi
import pickle
import os

if "MPICD_ROOT" in os.environ:
    prefix = os.environ["MPICD_ROOT"]
    mpih = f"{prefix}/include/mpi.h"
    libmpi = f"{prefix}/lib/libmpicd_capi.so"
else:
    topdir = os.path.dirname(os.path.dirname(__file__))
    mpih = f"{topdir}/mpicd-capi/include/mpi.h"
    libmpi = f"{topdir}/build/libmpicd_capi.so"

ffi = cffi.FFI()
with open(mpih) as fh:
    ffi.cdef("".join(list(fh)[9:-5]))

lib = ffi.dlopen(libmpi, ffi.RTLD_GLOBAL)
lib.MPI_Init(ffi.NULL, ffi.NULL)
atexit.register(lib.MPI_Finalize)


def CHKERR(ierr):
    if ierr:
        raise RuntimeError(ierr)


def pickle_dumps(obj):
    return pickle.dumps(obj, 5)


def pickle_loads(buf):
    return pickle.loads(buf)


def pickle_dumps_oob(obj):
    bufs = []
    data = pickle.dumps(obj, 5, buffer_callback=bufs.append)
    bufs.append(data)
    return bufs


def pickle_loads_oob(bufs):
    data = bufs.pop()
    return pickle.loads(data, buffers=bufs)


@ffi.callback('MPI_Type_custom_region_count_function')
def region_count_fn(
    state: 'void*',
    buf: 'void*',
    count: 'MPI_Count',
    region_count: 'MPI_Count',
):
    assert count == 1
    bufs = ffi.from_handle(buf)
    region_count[0] = len(bufs)
    return lib.MPI_SUCCESS


@ffi.callback('MPI_Type_custom_region_function')
def region_fn(
    state: 'void*',
    buf: 'void* ',
    count: 'MPI_Count',
    region_count: 'MPI_Count',
    reg_lens: 'MPI_Count[]',
    reg_bases: 'void*[]',
    types: 'MPI_Datatype[]',
):
    assert count == 1
    bufs = ffi.from_handle(buf)
    for i in range(region_count):
        b = ffi.from_buffer(bufs[i])
        reg_lens[i] = len(b)
        reg_bases[i] = b
        types[i] = lib.MPI_BYTE
    return lib.MPI_SUCCESS


def create_custom_datatype(bufs):
    NULL = ffi.NULL
    ctype = ffi.new('MPI_Datatype*')
    CHKERR( lib.MPI_Type_create_custom(
        NULL, NULL, NULL, NULL, NULL,
        region_count_fn, region_fn, NULL, 0, ctype) )
    return ctype[0]


def destroy_custom_datatype(ctype):
    ctype = ffi.new('MPI_Datatype*', ctype)
    pass # CHKERR( lib.MPI_Type_free(ctype) )


class Datatype:

    def __init__(self, handle):
        self.ob_mpi = handle

BYTE = Datatype(lib.MPI_BYTE)


ANY_SOURCE = lib.MPI_ANY_SOURCE
ANY_TAG = -2

class Intracomm:

    def __init__(self, handle):
        self.ob_mpi = handle

    def Get_size(self):
        size = ffi.new('int*')
        CHKERR( lib.MPI_Comm_size(self.ob_mpi, size) )
        return size[0]

    size = property(Get_size)

    def Get_rank(self):
        rank = ffi.new('int*')
        CHKERR( lib.MPI_Comm_rank(self.ob_mpi, rank) )
        return rank[0]

    rank = property(Get_rank)

    def Barrier(self):
        CHKERR( lib.MPI_Barrier(self.ob_mpi) )

    def Send(self, buf, dest, tag=0):
        comm = self.ob_mpi
        sbuf, scount, stype = buf
        sbuf = ffi.from_buffer(sbuf)
        stype = stype.ob_mpi
        CHKERR( lib.MPI_Send(sbuf, scount, stype, dest, tag, comm) )

    def Recv(self, buf, source=ANY_SOURCE, tag=ANY_TAG, status=None):
        assert status is None
        comm = self.ob_mpi
        rbuf, rcount, rtype = buf
        rbuf = ffi.from_buffer(rbuf)
        rtype = rtype.ob_mpi
        status = ffi.new('MPI_Status*')
        CHKERR( lib.MPI_Recv(rbuf, rcount, rtype, source, tag, comm, status) )

    def barrier(self):
        CHKERR( lib.MPI_Barrier(self.ob_mpi) )

    def send(self, obj, dest, tag=0):
        stype = lib.MPI_BYTE
        comm = self.ob_mpi
        buf = pickle_dumps(obj)
        sbuf = ffi.from_buffer(buf)
        scount = len(sbuf)
        CHKERR( lib.MPI_Send(sbuf, scount, stype, dest, tag, comm) )

    def recv(self, buf=None, source=ANY_SOURCE, tag=ANY_TAG, status=None):
        assert status is None
        rtype = lib.MPI_BYTE
        comm = self.ob_mpi
        status = ffi.new('MPI_Status*')
        nbytes = ffi.new('int*')
        CHKERR( lib.MPI_Probe(source, tag, comm, status) )
        CHKERR( lib.MPI_Get_count(status, rtype, nbytes) )
        nbytes = nbytes[0]
        source = status.MPI_SOURCE
        tag = status.MPI_TAG
        buf = bytearray(nbytes)
        rbuf = ffi.from_buffer(buf)
        rcount = len(rbuf)
        CHKERR( lib.MPI_Recv(rbuf, rcount, rtype, source, tag, comm, status) )
        return pickle_loads(buf)

    def send_oob(self, obj, dest, tag=0):
        comm = self.ob_mpi
        bufs = pickle_dumps_oob(obj)
        bufs = list(map(memoryview, bufs))
        lens = ffi.new('MPI_Count[]', list(map(len, bufs)))
        nbytes = len(lens) * ffi.sizeof('MPI_Count')
        CHKERR( lib.MPI_Send(lens, nbytes, lib.MPI_BYTE, dest, tag, comm) )
        sbuf = ffi.new_handle(bufs)
        stype = create_custom_datatype(bufs)
        CHKERR( lib.MPI_Send(sbuf, 1, stype, dest, tag, comm) )
        destroy_custom_datatype(stype)

    def recv_oob(self, buf=None, source=ANY_SOURCE, tag=ANY_TAG, status=None):
        assert status is None
        comm = self.ob_mpi
        status = ffi.new('MPI_Status*')
        CHKERR( lib.MPI_Probe(source, tag, comm, status) )
        source = status.MPI_SOURCE
        tag = status.MPI_TAG
        nbytes = ffi.new('int*')
        CHKERR( lib.MPI_Get_count(status, lib.MPI_BYTE, nbytes) )
        nbytes = nbytes[0]
        lens = ffi.new('MPI_Count[]', nbytes//ffi.sizeof('MPI_Count'))
        CHKERR( lib.MPI_Recv(lens, nbytes, lib.MPI_BYTE, source, tag, comm, status) )
        bufs = list(map(bytearray, lens))
        rbuf = ffi.new_handle(bufs)
        rtype = create_custom_datatype(bufs)
        CHKERR( lib.MPI_Recv(rbuf, 1, rtype, source, tag, comm, status) )
        destroy_custom_datatype(rtype)
        return pickle_loads_oob(bufs)

    def _isend(self, obj, dest, tag=0):
        stype = lib.MPI_BYTE
        comm = self.ob_mpi
        sbuf = pickle_dumps(obj)
        scount = len(sbuf)
        request = ffi.new('MPI_Request*')
        _sbuf = ffi.from_buffer(sbuf)
        CHKERR( lib.MPI_Isend(_sbuf, scount, stype, dest, tag, comm, request) )
        return request, sbuf

    def sendrecv(
        self,
        sendobj, dest, sendtag=0,
        recvbuf=None, source=ANY_SOURCE, recvtag=ANY_TAG,
        status=None,
    ):
        req, _ = self._isend(sendobj, dest, sendtag)
        obj = self.recv(recvbuf, source, recvtag, status)
        CHKERR( lib.MPI_Wait(req) )
        return obj

    def sendrecv_oob(*args, **kwargs):
        raise NotImplementedError


COMM_WORLD = Intracomm(lib.MPI_COMM_WORLD)
