#!/usr/bin/env python3

"""Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

# a timing script for FFTs and convolutions using OpenMP

import argparse
import subprocess
import os
import re # regexp package
import sys
import tempfile

def runcase(workingdir, mval, nval, kval, ntrial, precision, nbatch,
            devicenum, logfilename, function, side, uplo, diag, transA, transB, alpha, beta, incx, incy, lda, ldb, ldc, iters, cold_iters, algo):
    progname = "hipblas-bench"
    prog = os.path.join(workingdir, progname)

    cmd = []
    cmd.append(prog)

    cmd.append("-f")
    cmd.append(function)

    cmd.append("--transposeA")
    cmd.append(str(transA))

    cmd.append("--transposeB")
    cmd.append(str(transB))

    cmd.append("-m")
    cmd.append(str(mval))

    cmd.append("-n")
    cmd.append(str(nval))

    cmd.append("-k")
    cmd.append(str(kval))

    cmd.append("--side")
    cmd.append(str(side))

    cmd.append("--uplo")
    cmd.append(str(uplo))

    cmd.append("--diag")
    cmd.append(str(diag))

    cmd.append("--lda")
    cmd.append(str(lda))

    cmd.append("--ldb")
    cmd.append(str(ldb))

    cmd.append("--ldc")
    cmd.append(str(ldc))

    cmd.append("--alpha")
    cmd.append(str(alpha))

    cmd.append("--beta")
    cmd.append(str(beta))

    cmd.append("--incx")
    cmd.append(str(incx))

    cmd.append("--incy")
    cmd.append(str(incy))

    cmd.append("-i")
    cmd.append(str(iters))

    cmd.append("-j")
    cmd.append(str(cold_iters))

    cmd.append("-r")
    cmd.append(precision)

    cmd.append("--batch_count")
    cmd.append(str(nbatch))

    cmd.append("--algo")
    cmd.append(str(algo))

    cmd.append("--device")
    cmd.append(str(devicenum))

    print(" ".join(cmd))

    fout = tempfile.TemporaryFile(mode="w+")

    for x in range(ntrial):
        proc = subprocess.Popen(cmd, cwd=os.path.join(workingdir,"..",".."), stdout=fout, stderr=fout,
                                env=os.environ.copy())
        proc.wait()
        rc = proc.returncode

        if rc != 0:
            lines = fout.readlines()
            logfile = open(logfilename, "a")
            logfile.write('\n'.join(lines))
            logfile.close()
            print("\twell, that didn't work")
            print(rc)
            print(" ".join(cmd))
            return [], [], []

    us_vals = []
    gf_vals = []
    bw_vals = []

    fout.seek(0)

    lines = fout.readlines()
    logfile = open(logfilename, "a")
    logfile.write('\n'.join(lines))
    logfile.close()

    gf_string = "hipblas-Gflops"
    bw_string = "hipblas-GB/s"
    us_string = "us"
    for i in range(0, len(lines)):
        if re.search(r"\b" + re.escape(us_string) + r"\b", lines[i]) is not None:
            us_line = lines[i].strip().split(",")
            index = [idx for idx, s in enumerate(us_line) if us_string in s][0] #us_line.index()
            us_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
        if gf_string in lines[i]:
            gf_line = lines[i].split(",")
            index = gf_line.index(gf_string)
            gf_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
        if bw_string in lines[i]:
            bw_line = lines[i].split(",")
            index = bw_line.index(bw_string)
            bw_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))


    fout.close()

    return us_vals, gf_vals, bw_vals

def incrementParam(cur, max, mul, step_size, done):
    if cur < max:
        if mul == 1 and cur * step_size <= max:
            return cur*step_size, False
        elif mul != 1 and cur + step_size <= max:
            return cur+step_size, False
    return cur, done


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--workingdir',     required=False, default = '/home/marnauta/hipBLAS/build/release/clients/staging')
    parser.add_argument('-n', '--nmin',           required=False, default = 1, type=int)
    parser.add_argument('-N', '--nmax',           required=False, default = 1024, type=int)
    parser.add_argument('-k', '--kmin',           required=False, default = 1, type=int)
    parser.add_argument('-K', '--kmax',           required=False, default = 1, type=int)
    parser.add_argument('-m', '--mmin',           required=False, default = 1, type=int)
    parser.add_argument('-M', '--mmax',           required=False, default = 1, type=int)
    parser.add_argument('-a', '--ntrial',         required=False, default = 1, type=int)
    parser.add_argument('-i', '--iters',          required=False, default = 1, type=int)
    parser.add_argument('-j', '--cold_iters',     required=False, default = 2, type=int)
    parser.add_argument('-o', '--outfilename',    required=False, default = 'timing.dat')
    parser.add_argument('-p', '--precision',      required=False, default = 'f32_r')
    parser.add_argument('-b', '--nbatch',         required=False, default = 1, type=int)
    parser.add_argument('-r', '--radix',          required=False, default = 2, type=int)
    parser.add_argument('-s', '--step_size',      required=False, default = 10, type=int)
    parser.add_argument('-g', '--devicenum',      required=False, default = 0, type=int)
    parser.add_argument('-f', '--function',       required=False, default = '')
    parser.add_argument(      '--alpha',          required=False, default = 1, type=float)
    parser.add_argument(      '--beta',           required=False, default = 1, type=float)
    parser.add_argument(      '--incx',           required=False, default = 1, type=int)
    parser.add_argument(      '--incy',           required=False, default = 1, type=int)
    parser.add_argument(      '--transA',         required=False, default = 'N')
    parser.add_argument(      '--transB',         required=False, default = 'N')
    parser.add_argument(      '--lda',            required=False, default = 1, type=int)
    parser.add_argument(      '--ldb',            required=False, default = 1, type=int)
    parser.add_argument(      '--ldc',            required=False, default = 1, type=int)
    parser.add_argument(      '--LDA',            required=False, default = 1, type=int)
    parser.add_argument(      '--LDB',            required=False, default = 1, type=int)
    parser.add_argument(      '--LDC',            required=False, default = 1, type=int)
    parser.add_argument(      '--initialization', required=False, default = '')
    parser.add_argument('-x', '--step_mult',      required=False, default = False, action='store_true')
    parser.add_argument(      '--side',           required=False, default = 'L')
    parser.add_argument(      '--uplo',           required=False, default = 'L')
    parser.add_argument(      '--diag',           required=False, default = 'N')
    parser.add_argument(      '--algo',           required=False, default = 0, type=int)

    user_args = parser.parse_args()

    # de-namespace for backwards compatibility
    workingdir     = user_args.workingdir
    nmin           = user_args.nmin
    nmax           = user_args.nmax
    kmin           = user_args.kmin
    kmax           = user_args.kmax
    mmin           = user_args.mmin
    mmax           = user_args.mmax
    ntrial         = user_args.ntrial
    iters          = user_args.iters
    cold_iters     = user_args.cold_iters
    outfilename    = user_args.outfilename
    precision      = user_args.precision
    nbatch         = user_args.nbatch
    radix          = user_args.radix
    step_size      = user_args.step_size
    devicenum      = user_args.devicenum
    function       = user_args.function
    alpha          = user_args.alpha
    beta           = user_args.beta
    incx           = user_args.incx
    incy           = user_args.incy
    precision      = user_args.precision
    transA         = user_args.transA
    transB         = user_args.transB
    lda            = user_args.lda
    ldb            = user_args.ldb
    ldc            = user_args.ldc
    LDA            = user_args.LDA
    LDB            = user_args.LDB
    LDC            = user_args.LDC
    initialization = user_args.initialization
    step_mult      = 1 if user_args.step_mult else 0
    side           = user_args.side
    uplo           = user_args.uplo
    diag           = user_args.diag
    algo           = user_args.algo

    print("workingdir: "+ workingdir)
    print("outfilename: "+ outfilename)
    print("ntrial: " + str(ntrial))
    print("nmin: "+ str(nmin) + " nmax: " + str(nmax))
    print("batch-size: " + str(nbatch))
    # print("data type: " + datatype)
    print("device number: " + str(devicenum))

    progname = "hipblas-bench"
    prog = os.path.join(workingdir, progname)
    if not os.path.isfile(prog):
        print("**** Error: unable to find " + prog)
        sys.exit(1)

    logfilename = outfilename + ".log"
    print("log filename: "  + logfilename)
    logfile = open(logfilename, "w+")
    metadatastring = "# " + " ".join(sys.argv)  + "\n"
    logfile.write(metadatastring)
    logfile.close()

    outfile = open(outfilename, "w+")
    outfile.write(metadatastring)
    outfile.close()

    mval = mmin
    nval = nmin
    kval = kmin
    done = False
    while(not done):
        us, gf, bw = runcase(workingdir, mval, nval, kval, ntrial,
                          precision, nbatch, devicenum, logfilename, function, side, uplo, diag, transA, transB, alpha, beta, incx, incy, lda, ldb, ldc, iters, cold_iters, algo)
        #print(seconds)
        with open(outfilename, 'a') as outfile:
            if function == 'trsm':
                if side == 'R':
                    outfile.write(str(mval))
                else:
                    outfile.write(str(nval))
            else:
                outfile.write(str(nval))
            outfile.write("\t")
            outfile.write('{:4.0f}'.format(len(us)))
            for time in us:
                outfile.write('{:12.2f}'.format(time))
            outfile.write('{:4.0f}'.format(len(gf)))
            for flops in gf:
                outfile.write('{:10.2f}'.format(flops))
            outfile.write('{:4.0f}'.format(len(bw)))
            for bandwidth in bw:
                outfile.write('{:10.2f}'.format(bandwidth))
            outfile.write("\n")

        done = True
        mval, done = incrementParam(mval, mmax, step_mult, step_size, done)
        nval, done = incrementParam(nval, nmax, step_mult, step_size, done)
        kval, done = incrementParam(kval, kmax, step_mult, step_size, done)
        lda, done = incrementParam(lda, LDA, step_mult, step_size, done)
        ldb, done = incrementParam(ldb, LDB, step_mult, step_size, done)
        ldc, done = incrementParam(ldc, LDC, step_mult, step_size, done)



if __name__ == "__main__":
    main()
