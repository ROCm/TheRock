// Cross-process write -> mmap handoff probe for ROCm/TheRock#5024.
//
// The CI failure: one process (clang or ccache) writes an object file and
// exits, then a DIFFERENT process (lld-link) memory-maps it and sees all zeros.
// This reproduces exactly that shape at high volume so a ~1e-5/handoff fault is
// observable:
//
//   xproc_probe.exe run <root> <pairs> <iters> <sizeKB> <mode>
//     spawns <pairs> writer processes and <pairs> reader processes.
//   mode = inplace : open(CreateNew)+write+close at the final name (ccache
//   style) mode = rename  : write <name>.tmp, close, MoveFile to final name
//   (clang style)
//
// A writer signals completion by creating <name>.rdy only AFTER closing the
// data file (and renaming it, for mode=rename), so the reader never races a
// half-written file. The reader then maps the WHOLE file read-only
// (CreateFileMapping + MapViewOfFile, as LLVM MemoryBuffer does) and verifies
// every byte against the deterministic pattern. Any mismatch is a real
// coherency failure.
using System;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;

class P {
  static byte Pat(int id, int n, long i) {
    return (byte)((i * 131 + id * 7 + n * 13 + 1) % 251 + 1);
  }

  static int Main(string[] a) {
    if (a[0] == "run")
      return Run(a[1], int.Parse(a[2]), int.Parse(a[3]), int.Parse(a[4]), a[5]);
    if (a[0] == "w")
      return Writer(a[1], int.Parse(a[2]), int.Parse(a[3]), int.Parse(a[4]),
                    a[5]);
    if (a[0] == "r")
      return Reader(a[1], int.Parse(a[2]), int.Parse(a[3]), int.Parse(a[4]));
    if (a[0] == "w1")
      return WriteOne(a[1], int.Parse(a[2]), int.Parse(a[3]), int.Parse(a[4]),
                      a[5]);
    if (a[0] == "rw")
      return Rewrite(a[1], int.Parse(a[2]), a.Length > 3 ? a[3] : "plain");
    if (a[0] == "rw1")
      return Rewrite1(a[1], int.Parse(a[2]), int.Parse(a[3]),
                      a.Length > 4 ? a[4] : "plain");
    if (a[0] == "pr")
      return PreRead(a[1]);
    if (a[0] == "zs")
      return ZeroScan(a[1]);
    if (a[0] == "lld")
      return LldRun(a[1], int.Parse(a[2]), int.Parse(a[3]), int.Parse(a[4]),
                    int.Parse(a[5]));
    if (a[0] == "rb")
      return BatchReader(a[1], int.Parse(a[2]), int.Parse(a[3]),
                         int.Parse(a[4]));
    if (a[0] == "selftest")
      return SelfTest(a[1]);
    return 2;
  }

  static string F(string root, int id, int n) {
    return Path.Combine(root, "w" + id + "_" + n + ".bin");
  }

  // One file per process, like clang/ccache: the writer process EXITS before
  // the reader maps the file. Parent writes .rdy only after the child has
  // exited.
  static int WriteOne(string root, int id, int n, int sizeKB, string mode) {
    long size = sizeKB * 1024L;
    byte[] buf = new byte[size];
    for (long i = 0; i < size; i++)
      buf[i] = Pat(id, n, i);
    string fin = F(root, id, n);
    string target = mode == "spawnrename" ? fin + ".tmp" : fin;
    using (var fs = new FileStream(target, FileMode.CreateNew, FileAccess.Write,
                                   FileShare.None, 65536))
        fs.Write(buf, 0, buf.Length);
    if (mode == "spawnrename")
      File.Move(target, fin);
    return 0;
  }

  // lld-link shape: a burst of short-lived writer processes (ninja -jN compiles
  // / ccache restores; half write in place, half temp+rename) all exit, then
  // ONE fresh reader process maps every file concurrently, one thread per file,
  // the way lld/COFF createFutureForFile does with std::launch::async on Win64.
  static int LldRun(string root, int rounds, int batch, int sizeKB, int par) {
    string exe = Process.GetCurrentProcess().MainModule.FileName;
    long ok = 0, bad = 0;
    var sw = Stopwatch.StartNew();
    for (int r = 0; r < rounds; r++) {
      string dir = Path.Combine(root, "r" + r);
      Directory.CreateDirectory(dir);
      var running = new System.Collections.Generic.List<Process>();
      for (int i = 0; i < batch; i++) {
        while (running.Count >= par) {
          running[0].WaitForExit();
          running[0].Dispose();
          running.RemoveAt(0);
        }
        string mode = (i % 2 == 0) ? "spawn" : "spawnrename";
        running.Add(Process.Start(new ProcessStartInfo(
            exe, "w1 \"" + dir + "\" " + i + " " + r + " " + sizeKB + " " +
                     mode) { UseShellExecute = false, CreateNoWindow = true }));
      }
      foreach (var p in running) {
        p.WaitForExit();
        p.Dispose();
      }
      var rp = Process.Start(new ProcessStartInfo(
          exe, "rb \"" + dir + "\" " + r + " " + batch + " " +
                   sizeKB) { UseShellExecute = false,
                             RedirectStandardOutput = true,
                             CreateNoWindow = true });
      string o = rp.StandardOutput.ReadToEnd();
      rp.WaitForExit();
      foreach (var line in o.Split('\n')) {
        string l = line.Trim();
        if (l.StartsWith("BAD")) {
          bad++;
          Console.WriteLine(root + " round=" + r + " " + l);
        } else if (l.StartsWith("OKCOUNT="))
          ok += long.Parse(l.Substring(8));
      }
      try {
        Directory.Delete(dir, true);
      } catch {
      }
    }
    Console.WriteLine(
        "RESULT root=" + root + " mode=lld handoffs=" + (ok + bad) +
        " ok=" + ok + " bad=" + bad + " rounds=" + rounds +
        " batch=" + batch + " secs=" + (int)sw.Elapsed.TotalSeconds);
    return 0;
  }

  static int BatchReader(string dir, int n, int batch, int sizeKB) {
    long size = sizeKB * 1024L;
    int ok = 0;
    var threads = new Thread[batch];
    var msgs = new string[batch];
    for (int i = 0; i < batch; i++) {
      int id = i;
      threads[i] = new Thread(() => {
        try {
          string d;
          int r = Check(F(dir, id, n), id, n, size, out d);
          if (r == 0)
            Interlocked.Increment(ref ok);
          else {
            string d2;
            int r2 = Check(F(dir, id, n), id, n, size, out d2);
            msgs[id] = "BAD kind=" + r + " id=" + id + " " + d +
                       " | recheck=" + r2 + " " + d2;
          }
        } catch (Exception e) {
          msgs[id] =
              "BAD kind=9 id=" + id + " " + e.GetType().Name + ": " + e.Message;
        }
      });
    }
    foreach (var t in threads)
      t.Start();
    foreach (var t in threads)
      t.Join();
    foreach (var m in msgs)
      if (m != null)
        Console.WriteLine(m);
    Console.WriteLine("OKCOUNT=" + ok);
    return 0;
  }

  // For every file in <listfile>, report whether its first 64 bytes are all
  // zero when read via a fresh read-only mapping (lld's path) and via ReadFile.
  static int ZeroScan(string list) {
    int bad = 0, n = 0;
    foreach (var raw in File.ReadAllLines(list)) {
      string f = raw.Trim();
      if (f.Length == 0)
        continue;
      n++;
      bool mz = false, rz = false;
      long len = new FileInfo(f).Length;
      int k = (int)Math.Min(64, len);
      using (var mmf = MemoryMappedFile.CreateFromFile(
                 new FileStream(f, FileMode.Open, FileAccess.Read,
                                FileShare.ReadWrite | FileShare.Delete),
                 null, 0, MemoryMappedFileAccess.Read,
                 HandleInheritability.None,
                 false)) using (var v =
                                    mmf.CreateViewAccessor(
                                        0, k, MemoryMappedFileAccess.Read)) {
        mz = true;
        for (int i = 0; i < k; i++)
          if (v.ReadByte(i) != 0) {
            mz = false;
            break;
          }
      }
      var b = new byte[k];
      using (var fs = new FileStream(f, FileMode.Open, FileAccess.Read,
                                     FileShare.ReadWrite | FileShare.Delete, 1,
                                     FileOptions.None)) fs.Read(b, 0, k);
      rz = true;
      for (int i = 0; i < k; i++)
        if (b[i] != 0) {
          rz = false;
          break;
        }
      if (mz || rz) {
        bad++;
        Console.WriteLine("ZERO mmap=" + mz + " read=" + rz + " len=" + len +
                          " " + f);
      }
    }
    Console.WriteLine("ZSCAN files=" + n + " zero=" + bad);
    return 0;
  }

  // Rewrite every file in <listfile> in place (truncate + write identical
  // bytes), split across <par> short-lived child processes, like ccache
  // restoring objects.
  static int Rewrite(string list, int par, string mode) {
    string exe = Process.GetCurrentProcess().MainModule.FileName;
    var ps = new Process[par];
    for (int i = 0; i < par; i++)
      ps[i] = Process.Start(new ProcessStartInfo(
          exe, "rw1 \"" + list + "\" " + i + " " + par + " " +
                   mode) { UseShellExecute = false, CreateNoWindow = true });
    int rc = 0;
    foreach (var p in ps) {
      p.WaitForExit();
      rc |= p.ExitCode;
      p.Dispose();
    }
    return rc;
  }

  // mode: plain = truncate+write in place; flush = plain + FlushFileBuffers;
  // rename = write temp then atomic replace.
  static int Rewrite1(string list, int idx, int par, string mode) {
    var files = File.ReadAllLines(list);
    for (int i = idx; i < files.Length; i += par) {
      string f = files[i].Trim();
      if (f.Length == 0)
        continue;
      byte[] b = File.ReadAllBytes(f);
      if (mode == "rename") {
        string t = f + ".tmp" + idx;
        using (var fs = new FileStream(t, FileMode.Create, FileAccess.Write,
                                       FileShare.None))
            fs.Write(b, 0, b.Length);
        File.Delete(f);
        File.Move(t, f);
        continue;
      }
      using (var fs = new FileStream(f, FileMode.Create, FileAccess.Write,
                                     FileShare.Read | FileShare.Delete)) {
        fs.Write(b, 0, b.Length);
        if (mode == "flush")
          fs.Flush(true);
      }
    }
    return 0;
  }

  // Read every file fully with ReadFile (warm the cache through the non-mmap
  // path).
  static int PreRead(string list) {
    long tot = 0;
    foreach (var raw in File.ReadAllLines(list)) {
      string f = raw.Trim();
      if (f.Length > 0)
        tot += File.ReadAllBytes(f).Length;
    }
    return tot > 0 ? 0 : 1;
  }

  static int Writer(string root, int id, int iters, int sizeKB, string mode) {
    long size = sizeKB * 1024L;
    if (mode.StartsWith("spawn")) {
      string exe = Process.GetCurrentProcess().MainModule.FileName;
      for (int n = 0; n < iters; n++) {
        var psi = new ProcessStartInfo(
            exe, "w1 \"" + root + "\" " + id + " " + n + " " + sizeKB + " " +
                     mode) { UseShellExecute = false, CreateNoWindow = true };
        using (var p = Process.Start(psi)) p.WaitForExit();
        File.WriteAllBytes(F(root, id, n) + ".rdy", new byte[] { 1 });
      }
      return 0;
    }
    for (int n = 0; n < iters; n++) {
      byte[] buf = new byte[size];
      for (long i = 0; i < size; i++)
        buf[i] = Pat(id, n, i);
      string fin = F(root, id, n);
      string target = mode == "rename" ? fin + ".tmp" : fin;
      using (var fs = new FileStream(target, FileMode.CreateNew,
                                     FileAccess.Write, FileShare.None, 65536))
          fs.Write(buf, 0, buf.Length);
      if (mode == "rename")
        File.Move(target, fin);
      File.WriteAllBytes(fin + ".rdy", new byte[] { 1 });
    }
    return 0;
  }

  // Returns: 0 ok, 1 all-zero, 2 other mismatch, 3 short
  static int Check(string path, int id, int n, long size, out string detail) {
    detail = "";
    var fi = new FileInfo(path);
    if (fi.Length != size) {
      detail = "len=" + fi.Length;
      return 3;
    }
    using (
        var fs = new FileStream(
            path, FileMode.Open, FileAccess.Read,
            FileShare.ReadWrite |
                FileShare
                    .Delete)) using (var mmf =
                                         MemoryMappedFile.CreateFromFile(
                                             fs, null, 0,
                                             MemoryMappedFileAccess.Read, null,
                                             HandleInheritability.None,
                                             true)) using (var v =
                                                               mmf.CreateViewAccessor(
                                                                   0, 0,
                                                                   MemoryMappedFileAccess
                                                                       .Read)) {
      byte[] got = new byte[size];
      v.ReadArray(0, got, 0, (int)size);
      long firstBad = -1, zeros = 0;
      for (long i = 0; i < size; i++) {
        if (got[i] == 0)
          zeros++;
        if (firstBad < 0 && got[i] != Pat(id, n, i))
          firstBad = i;
      }
      if (firstBad < 0)
        return 0;
      detail = "firstBad=" + firstBad + " zeroBytes=" + zeros + "/" + size;
      return zeros == size ? 1 : 2;
    }
  }

  static int Reader(string root, int id, int iters, int sizeKB) {
    long size = sizeKB * 1024L;
    int ok = 0, zero = 0, mism = 0, shrt = 0, timeout = 0;
    for (int n = 0; n < iters; n++) {
      string fin = F(root, id, n);
      var sw = Stopwatch.StartNew();
      while (!File.Exists(fin + ".rdy")) {
        if (sw.ElapsedMilliseconds > 120000)
          break;
        Thread.SpinWait(200);
      }
      if (!File.Exists(fin + ".rdy")) {
        timeout++;
        Console.WriteLine("TIMEOUT id=" + id + " n=" + n);
        continue;
      }
      string d;
      int r = Check(fin, id, n, size, out d);
      if (r == 0)
        ok++;
      else {
        if (r == 1)
          zero++;
        else if (r == 2)
          mism++;
        else
          shrt++;
        string d2;
        int r2 = Check(fin, id, n, size, out d2);
        Console.WriteLine("BAD kind=" + r + " id=" + id + " n=" + n + " " + d +
                          " | recheck=" + r2 + " " + d2);
      }
      try {
        File.Delete(fin);
        File.Delete(fin + ".rdy");
      } catch {
      }
    }
    Console.WriteLine("READER id=" + id + " ok=" + ok + " zero=" + zero +
                      " mism=" + mism + " short=" + shrt +
                      " timeout=" + timeout);
    return 0;
  }

  static int Run(string root, int pairs, int iters, int sizeKB, string mode) {
    if (Directory.Exists(root))
      Directory.Delete(root, true);
    Directory.CreateDirectory(root);
    string exe = Process.GetCurrentProcess().MainModule.FileName;
    var sw = Stopwatch.StartNew();
    var readers = new Process[pairs];
    var writers = new Process[pairs];
    for (int id = 0; id < pairs; id++) {
      readers[id] =
          Start(exe, "r \"" + root + "\" " + id + " " + iters + " " + sizeKB);
      writers[id] = Start(exe, "w \"" + root + "\" " + id + " " + iters + " " +
                                   sizeKB + " " + mode);
    }
    long ok = 0, zero = 0, mism = 0, shrt = 0, to = 0;
    foreach (var p in readers) {
      string outp = p.StandardOutput.ReadToEnd();
      p.WaitForExit();
      foreach (var line in outp.Split('\n')) {
        string l = line.Trim();
        if (l.StartsWith("BAD") || l.StartsWith("TIMEOUT"))
          Console.WriteLine(root + " " + l);
        if (l.StartsWith("READER"))
          foreach (var kv in l.Split(' ')) {
            var s = kv.Split('=');
            if (s.Length != 2)
              continue;
            long x = long.Parse(s[1]);
            if (s[0] == "ok")
              ok += x;
            else if (s[0] == "zero")
              zero += x;
            else if (s[0] == "mism")
              mism += x;
            else if (s[0] == "short")
              shrt += x;
            else if (s[0] == "timeout")
              to += x;
          }
      }
    }
    foreach (var p in writers)
      p.WaitForExit();
    Console.WriteLine("RESULT root=" + root + " mode=" + mode +
                      " handoffs=" + (ok + zero + mism + shrt) + " ok=" + ok +
                      " zero=" + zero + " mism=" + mism + " short=" + shrt +
                      " timeout=" + to +
                      " secs=" + (int)sw.Elapsed.TotalSeconds);
    try {
      Directory.Delete(root, true);
    } catch {
    }
    return 0;
  }

  static Process Start(string exe, string args) {
    var psi = new ProcessStartInfo(exe, args) { UseShellExecute = false,
                                                RedirectStandardOutput = true,
                                                CreateNoWindow = true };
    return Process.Start(psi);
  }

  // Negative control: a zero-filled file of the right size MUST be flagged
  // kind=1.
  static int SelfTest(string root) {
    Directory.CreateDirectory(root);
    string f = F(root, 0, 0);
    File.WriteAllBytes(f, new byte[65536]);
    string d;
    int r = Check(f, 0, 0, 65536, out d);
    File.Delete(f);
    byte[] good = new byte[65536];
    for (int i = 0; i < good.Length; i++)
      good[i] = Pat(0, 0, i);
    File.WriteAllBytes(f, good);
    string d2;
    int r2 = Check(f, 0, 0, 65536, out d2);
    File.Delete(f);
    Console.WriteLine("SELFTEST zero_detected=" + (r == 1) +
                      " good_ok=" + (r2 == 0));
    return (r == 1 && r2 == 0) ? 0 : 1;
  }
}
