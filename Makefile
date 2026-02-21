make
.PHONY: all therock configure build

all: therock

configure:
	python3 build_tools/ninja_jobserver/build_the_rock.py configure

build:
	python3 build_tools/ninja_jobserver/build_the_rock.py build

therock:
	python3 build_tools/ninja_jobserver/build_the_rock.py all
