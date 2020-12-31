# Face Authentication Wrapper

This project is a GO-C++ wrapper around the DLib project.
The goal is to be able to authenticate 2 persons.


## Build
Built on ubuntu
`sudo apt-get libdlib-dev libblas-dev liblapack-dev libjpeg-turbo8-dev golang-go swig`


```export CGO_CXXFLAGS="-ldlib -lstdc++ -std=c++11 -lblas -lm -llapack -lsass"
export CGO_LDFLAGS="-llapack -ldlib -lblas -ljpeg"
go build -x
go install -x```

Import this project in your go project !