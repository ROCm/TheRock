#!/bin/bash

ROCBLAS_BENCH=../../../build/release/clients/staging/hipblas-bench
TAG1=gfx90a
TAG2=ref
LEVEL1=true
LEVEL2=false
LEVEL3=false
BENCHMARK=true
PLOT=true
VS_THEO_MAX=false
VS_PERF=false

usage()
{
    echo ""
    echo "Usage: $0 --tag1 <tag1>  -b <hipblas_bench> <--plot> <--benchmark> <--level1> <--level2> <--level3> <--theo_max> <--perf_vs_perf>"
    echo ""
    echo "where tag1 = tag for storing files, typically set to architecture like: gfx906, gfx90a, ..."
    echo "            default: $TAG1"
    echo ""
    echo "where tag2 = tag for storing files with reference performance, typically set to architecture like: gfx906, gfx90a, ..."
    echo "            default: $TAG2"
    echo ""
    echo "where hipblas_bench = path to hipblas_bench"
    echo "                      default: $ROCBLAS_BENCH"
    echo ""
    echo "--benchmark true:  to run benchmarks"
    echo "                   default: $BENCHMARK"
    echo ""
    echo "--plot true:  to plot results"
    echo "              default: $PLOT"
    echo ""
    echo "--level1 true:  for L1 BLAS"
    echo "                default: $LEVEL1"
    echo ""
    echo "--level2 true:  for L2 BLAS"
    echo "                default: $LEVEL2"
    echo ""
    echo "--level3 true:  for L3 BLAS"
    echo "                default: $LEVEL3"
    echo ""
    echo "--theo-max true:  to plot performance / theoretical_maximum_performance,"
    echo "                  only for BLAS1 and BLAS2, not for BLAS3"
    echo "                  default: $VS_THEO_MAX"
    echo ""
    echo "--perf_vs_perf true:  to plot performance / performance,"
    echo "                  default: $VS_PERF"
    echo ""
    echo "Example: $0 --tag1 gfx90a --plot false --benchmark true --level1 false --level2 false --level3 true"
    exit 1

}

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -b)
      ROCBLAS_BENCH="$2"
      shift # past argument
      shift # past value
      ;;
    --tag1)
      TAG1="$2"
      shift # past argument
      shift # past value
      ;;
    --tag2)
      TAG2="$2"
      shift # past argument
      shift # past value
      ;;
    --level1)
      LEVEL1="$2"
      shift # past argument
      shift # past value
      ;;
    --level2)
      LEVEL2="$2"
      shift # past argument
      shift # past value
      ;;
    --level3)
      LEVEL3="$2"
      shift # past argument
      shift # past value
      ;;
    --benchmark)
      BENCHMARK="$2"
      shift # past argument
      shift # past value
      ;;
    --plot)
      PLOT="$2"
      shift # past argument
      shift # past value
      ;;
    --theo_max)
      VS_THEO_MAX="$2"
      shift # past argument
      shift # past value
      ;;
    --perf_vs_perf)
      VS_PERF="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      usage
      ;;
    -*|--*)
      echo "Unknown option $1"
      usage
      ;;
    *)
      echo "unknown argument"
      usage
      ;;
  esac
done

if [ "$BENCHMARK" == "true" ] &&  [ ! -f $ROCBLAS_BENCH ]
then
    echo "the following file does not exist: $ROCBLAS_BENCH"
    echo "specify path to hipblas-bench"
    usage
    exit 1
fi

if [ "$BENCHMARK" == "true" ]; then
    echo "hipblas-bench path = $ROCBLAS_BENCH"
fi
echo "tag1               = $TAG1"
echo "tag2               = $TAG2"
echo "theo_max           = $VS_THEO_MAX"
echo "perf_vs_perf       = $VS_PERF"
echo ""
echo "benchmark          = $BENCHMARK"
echo "plot               = $PLOT"
echo ""
echo "level 1            = $LEVEL1"
echo "level 2            = $LEVEL2"
echo "level 3            = $LEVEL3"
echo ""


if [ "$VS_THEO_MAX" == "true" ] && [ "$VS_PERF" == "true" ]; then
    echo "select either theo_max or perf_vs_perf but not both"
    exit 1
fi

if [ "$VS_THEO_MAX" == "true" ]; then
    THEO_MAX="--theo_max"
else
    THEO_MAX=""
fi


if [ "$VS_PERF" == "true" ]; then
    PERF_VS_PERF="--perf_vs_perf"
else
    PERF_VS_PERF=""
fi

if [ "$LEVEL1" == "true" ]; then
  if [ "$BENCHMARK" == "true" ]; then
    python3 benchmark.py -l blas1 -t $TAG1 -b $ROCBLAS_BENCH -f dot -f axpy -f scal -f nrm2
  fi
  if [ "$PLOT" == "true" ]; then
    python3 plot.py -l blas1 --tag1 $TAG1 --tag2  $TAG2 $THEO_MAX $PERF_VS_PERF -f dot -f axpy -f scal
  fi
fi

if [ "$LEVEL2" == "true" ]; then
  if [ "$BENCHMARK" == "true" ]; then
    python3 benchmark.py -l blas2 -t $TAG1 -b $ROCBLAS_BENCH -f hemv -f her2 -f hpr2 -f her -f hpr -f hpmv -f hbmv
    python3 benchmark.py -l blas2 -t $TAG1 -b $ROCBLAS_BENCH -f gemv -f symv -f trmv -f tpmv -f spmv -f gbmv -f sbmv -f tbmv
    python3 benchmark.py -l blas2 -t $TAG1 -b $ROCBLAS_BENCH -f ger -f syr -f spr -f syr2 -f spr2
    python3 benchmark.py -l blas2 -t $TAG1 -b $ROCBLAS_BENCH -f trsv -f tbsv -f tpsv
  fi
  if [ "$PLOT" == "true" ]; then
    python3 plot.py -l blas2 --tag1 $TAG1 --tag2 $TAG2 $THEO_MAX $PERF_VS_PERF --label1 "M" --label2 "N" -f hemv -f her2 -f hpr2 -f her -f hpr -f hpmv
    python3 plot.py -l blas2 --tag1 $TAG1 --tag2 $TAG2 $THEO_MAX $PERF_VS_PERF --label1 "M" --label2 "N" -f gemv -f trmv -f tpmv -f tbmv # missing symv, spmv, gbmv, sbmv
    python3 plot.py -l blas2 --tag1 $TAG1 --tag2 $TAG2 $THEO_MAX $PERF_VS_PERF --label1 "M" --label2 "N" -f ger -f syr -f spr -f syr2 -f spr2
#   python3 plot.py -l blas2 --tag1 $TAG1 --tag2 $TAG2 $THEO_MAX $PERF_VS_PERF --label1 "M" --label2 "N" -f trsv -f tpsv -f tbsv # need to speed up triangle solve initialization
  fi
fi

if [ "$LEVEL3" == "true" ]; then
  if [ "$THEO_MAX" = "--theo_max" ]; then
    echo "--theo-max is not for BLAS3, it can only be used with BLAS1 and BLAS2"
    echo "           for BLAS3 compare performance versus gemm"
    exit
  fi

  if [ "$BENCHMARK" == "true" ]; then
    python3 benchmark.py -l blas3 -t $TAG1 -b $ROCBLAS_BENCH -f gemm
    python3 benchmark.py -l blas3 -t $TAG1 -b $ROCBLAS_BENCH -f trmm -f symm -f trsm
    python3 benchmark.py -l blas3 -t $TAG1 -b $ROCBLAS_BENCH -f syrk -f syrkx -f syr2k
    python3 benchmark.py -l blas3 -t $TAG1 -b $ROCBLAS_BENCH -f hemm -f herk -f herkx -f her2k
  fi
  if [ "$PLOT" == "true" ]; then
    python3 plot.py -l blas3 --tag1 $TAG1 --tag2 $TAG2 $THEO_MAX $PERF_VS_PERF --label1 "M" --label2 "N" -f gemm -f trmm -f symm # missing trsm, need to speed up triangle solve initialization
    python3 plot.py -l blas3 --tag1 $TAG1 --tag2 $TAG2 $THEO_MAX $PERF_VS_PERF --label1 "M" --label2 "N" -f gemm -f syrk -f syrkx -f syr2k
    python3 plot.py -l blas3 --tag1 $TAG1 --tag2 $TAG2 $THEO_MAX $PERF_VS_PERF --label1 "M" --label2 "N" -f gemm -f hemm -f herk -f herkx -f her2k
  fi
fi
