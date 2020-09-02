SOURCE = src/main.py

VERSION := $(shell poetry version | grep -oE '[^ ]+$$')

PLATFORM :=
ifeq ($(OS),Windows_NT)
	PLATFORM = win32
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		PLATFORM = linux
	endif
	ifeq ($(UNAME_S),Darwin)
		PLATFORM = darwin
	endif
endif

FILENAME = PROJECT_NAME-$(VERSION)-$(PLATFORM)

BUILDFLAGS = --onefile --name $(FILENAME)

.PHONY = all list clean release build version

all: list

list:
	@sh -c "$(MAKE) -p no_targets__ | \
		awk -F':' '/^[a-zA-Z0-9][^\$$#\/\\t=]*:([^=]|$$)/ {\
			split(\$$1,A,/ /);for(i in A)print A[i]\
		}' | grep -v '__\$$' | grep -v 'make\[1\]' | grep -v 'Makefile' | sort"

clean:
	rm -rf build/ dist/
	rm -f *.spec
	@poetry run pyclean .

release:
	git tag v$(VERSION)
	git push origin v$(VERSION)

build:
	@poetry run pyinstaller $(BUILDFLAGS) $(SOURCE)

version:
	@echo $(VERSION)
