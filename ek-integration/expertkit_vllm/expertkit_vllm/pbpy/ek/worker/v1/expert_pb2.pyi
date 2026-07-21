from ek.object.v1 import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ForwardReq(_message.Message):
    __slots__ = ("instance_id", "sequences", "tensor", "request_id", "microbatch_id", "layer_id", "pipeline_enabled")
    class SequenceInfo(_message.Message):
        __slots__ = ("experts",)
        EXPERTS_FIELD_NUMBER: _ClassVar[int]
        experts: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, experts: _Optional[_Iterable[str]] = ...) -> None: ...
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MICROBATCH_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_ENABLED_FIELD_NUMBER: _ClassVar[int]
    instance_id: str
    sequences: _containers.RepeatedCompositeFieldContainer[ForwardReq.SequenceInfo]
    tensor: bytes
    request_id: str
    microbatch_id: int
    layer_id: int
    pipeline_enabled: bool
    def __init__(self, instance_id: _Optional[str] = ..., sequences: _Optional[_Iterable[_Union[ForwardReq.SequenceInfo, _Mapping]]] = ..., tensor: _Optional[bytes] = ..., request_id: _Optional[str] = ..., microbatch_id: _Optional[int] = ..., layer_id: _Optional[int] = ..., pipeline_enabled: bool = ...) -> None: ...

class ForwardResp(_message.Message):
    __slots__ = ("output_tensor", "request_id", "microbatch_id", "layer_id")
    OUTPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MICROBATCH_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    output_tensor: bytes
    request_id: str
    microbatch_id: int
    layer_id: int
    def __init__(self, output_tensor: _Optional[bytes] = ..., request_id: _Optional[str] = ..., microbatch_id: _Optional[int] = ..., layer_id: _Optional[int] = ...) -> None: ...

class ExpertState(_message.Message):
    __slots__ = ("stage",)
    class Stage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STAGE_UNSPECIFIED: _ClassVar[ExpertState.Stage]
        STAGE_ACTIVE: _ClassVar[ExpertState.Stage]
        STAGE_LOADING: _ClassVar[ExpertState.Stage]
        STAGE_EVICTING: _ClassVar[ExpertState.Stage]
    STAGE_UNSPECIFIED: ExpertState.Stage
    STAGE_ACTIVE: ExpertState.Stage
    STAGE_LOADING: ExpertState.Stage
    STAGE_EVICTING: ExpertState.Stage
    STAGE_FIELD_NUMBER: _ClassVar[int]
    stage: ExpertState.Stage
    def __init__(self, stage: _Optional[_Union[ExpertState.Stage, str]] = ...) -> None: ...

class ExchangeReq(_message.Message):
    __slots__ = ("id", "addr", "channel", "device", "last_will", "rdma_tcp_port")
    ID_FIELD_NUMBER: _ClassVar[int]
    ADDR_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    LAST_WILL_FIELD_NUMBER: _ClassVar[int]
    RDMA_TCP_PORT_FIELD_NUMBER: _ClassVar[int]
    id: str
    addr: str
    channel: str
    device: str
    last_will: bool
    rdma_tcp_port: int
    def __init__(self, id: _Optional[str] = ..., addr: _Optional[str] = ..., channel: _Optional[str] = ..., device: _Optional[str] = ..., last_will: bool = ..., rdma_tcp_port: _Optional[int] = ...) -> None: ...

class ExchangeResp(_message.Message):
    __slots__ = ("state",)
    class ExpertWithState(_message.Message):
        __slots__ = ("target",)
        TARGET_FIELD_NUMBER: _ClassVar[int]
        target: _object_pb2.ExpertSlice
        def __init__(self, target: _Optional[_Union[_object_pb2.ExpertSlice, _Mapping]] = ...) -> None: ...
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: ExchangeResp.ExpertWithState
    def __init__(self, state: _Optional[_Union[ExchangeResp.ExpertWithState, _Mapping]] = ...) -> None: ...
