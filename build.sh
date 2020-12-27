export CGO_CXXFLAGS="-ldlib -lstdc++ -std=c++11 -lblas -lm -llapack -lsass"
export CGO_LDFLAGS="-llapack -ldlib -lblas"
go build -x
go install -x
