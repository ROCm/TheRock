KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	sudo insmod amdgpu_lite.ko

uninstall:
	sudo rmmod amdgpu_lite

.PHONY: all clean install uninstall
