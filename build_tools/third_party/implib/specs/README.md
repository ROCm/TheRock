This directory contains formal models of Implib.so algorithm
which should help with verification.

# Initialization

This is a model of Implib.so initialization logic with `-DIMPLIB_EXPORT_SHIMS`
and is based on arch/common/init.c.tpl.

The model is implemented in two flavors: Promela and TLA+
(I have no idea why I didn't use PlusCal).

To verify TLA+ model
- download tla2tools.jar from https://github.com/tlaplus/tlaplus
- execute
```
$ rm -rf states
$ java -jar ./tla2tools.jar -workers $(nproc) Init.tla
```
(add `-coverage 1` for coverage stats).

To verify Promela model
- install spin (via `sudo apt install spin`)
- execute
```
$ spin -run -ltl never_0 Init.pml
$ spin -run -ltl Prop Init.pml
```
Errors can be examined via
```
$ spin -p -t Init.pml
```

TODO:
- try other langs (Alloy, SMV, B)
- integrate TLA model with Apalache
