# ======================================================================================================
#
#   Defines some unique/singleton for check status.
#
# ======================================================================================================


from abc import ABC


class _StatusType(ABC):
    __slots__ = ()

    __TYPE__ = ""

    def __str__(self): ...

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool: ...

    def __eq__(self, other):
        # 只要大家都有 __TYPE__ 且字串一樣，就當作相等！
        if hasattr(other, "__TYPE__"):
            return self.__TYPE__ == other.__TYPE__
        return False

    def __hash__(self):
        return hash(self.__TYPE__)


#   ====================================================================================================


#   Not Found Type.
#   We use this for file not found status.
#   On logics see it as False.


class _NotFoundType(_StatusType):
    __slots__ = ()

    def __str__(self):
        return "NOTFOUND"

    def __repr__(self) -> str:
        return "NOTFOUND"

    def __bool__(self) -> bool:
        return False

    def __format__(self, format_spec):
        return format(str(self), format_spec)

    def as_posix(self):
        return self.__str__()

    def split(self):
        return self.__str__()


NOTFOUND = _NotFoundType()
NotFoundType = type(NOTFOUND)


#   Not Defined Type.
#   We use this for some value not defined status.
#   On logics see it as False.


class _NotDefinedType(_StatusType):
    __slots__ = ()

    def __str__(self):
        return "NOTDEFINED"

    def __repr__(self):
        return "NOTDEFINED Type"

    def __bool__(self):
        return False

    def __format__(self, format_spec):
        return format(str(self), format_spec)

    def split(self):
        return self.__str__()

    def strip(self):
        return self.__str__()

    def parent(self):
        return self


NOTDEFINED = _NotDefinedType()
NotDefinedType = type(NOTDEFINED)


#   Pass Type.
#   We see this is a successful search, pass/success type.


class _SuccessType(_StatusType):
    __slots__ = ()

    __TYPE__ = "SUCCESS"

    def __str__(self):
        from .color_string import cstring

        return cstring(self.__TYPE__, "SUCCESS").__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return True


SUCCESS = _SuccessType()
SuccessType = type(SUCCESS)


#   Hint Type.
#   We see this is a successful search, pass/success type, with notice telling user.


class _HintType(_StatusType):
    __slots__ = ()

    __TYPE__ = "HINT"

    def __str__(self) -> str:
        from .color_string import cstring

        return cstring(self.__TYPE__, "HINT").__str__()

    def __bool__(self) -> str:
        return True


HINT = _HintType()
HintType = type(HINT)


#   Failed Type.
#   We see this is a successful search, failed type, not fatal effection can ignores.


class _FailedType(_StatusType):
    __slots__ = ()

    __TYPE__ = "FAILED"

    def __repr__(self):
        return "FAILED"

    def __bool__(self):
        return False


FAILED = _FailedType()
FailedType = type(FAILED)


#   Fatal Type.
#   We see this is a successful search, fatal type, a fatal effection cannot ignores.


class _FatalType(_StatusType):
    __slots__ = ()

    __TYPE__ = "FATAL"

    def __str__(self):
        from .color_string import cstring

        return cstring(self.__TYPE__, "ERROR").__str__()

    def __bool__(self):
        return False


FATAL = _FatalType()
FatalType = type(FATAL)


#   Warning Type.
#   We see this is a successful search, success type, but will effection cannot ignores.


class _WarningType(_StatusType):
    __slots__ = ()

    __TYPE__ = "WARNING"

    def __str__(self):
        from .color_string import cstring

        return cstring(self.__TYPE__, "WARNING").__str__()

    def __bool__(self):
        return True


WARNING = _WarningType()
WarningType = type(WARNING)
