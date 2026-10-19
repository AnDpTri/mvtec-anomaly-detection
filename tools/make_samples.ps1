# Ghép ảnh mẫu MVTec AD: mỗi hàng là một kiểu lỗi, mỗi mẫu gồm ảnh gốc và ảnh có vùng lỗi tô đỏ.
# Chỉ dùng System.Drawing có sẵn trong Windows, không cần cài thư viện.
param(
    [string]$Root = "D:\DeTai22\datasets\MVTecAD",
    [string]$OutDir = "D:\DeTai22\results\samples",
    [string[]]$Categories = @("bottle", "screw"),
    [int]$PerType = 2,
    [int]$Cell = 300
)

Add-Type -AssemblyName System.Drawing
Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @"
using System.Drawing;
using System.Drawing.Imaging;
public static class Overlay {
    // Tô đỏ bán trong suốt những pixel mà mask trắng, và vẽ viền đỏ đậm ở mép vùng lỗi.
    public static Bitmap Apply(Bitmap img, Bitmap mask) {
        int w = img.Width, h = img.Height;
        var outBmp = new Bitmap(img);
        var m = new Bitmap(mask, w, h);
        bool[,] on = new bool[w, h];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) on[x, y] = m.GetPixel(x, y).R > 127;
        var r = new Rectangle(0, 0, w, h);
        var d = outBmp.LockBits(r, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
        byte[] buf = new byte[d.Stride * h];
        System.Runtime.InteropServices.Marshal.Copy(d.Scan0, buf, 0, buf.Length);
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            if (!on[x, y]) continue;
            int i = y * d.Stride + x * 4;
            bool edge = x == 0 || y == 0 || x == w - 1 || y == h - 1 ||
                        !on[x - 1, y] || !on[x + 1, y] || !on[x, y - 1] || !on[x, y + 1];
            if (edge) { buf[i] = 0; buf[i + 1] = 0; buf[i + 2] = 255; }
            else { buf[i] = (byte)(buf[i] * 0.55); buf[i + 1] = (byte)(buf[i + 1] * 0.55); buf[i + 2] = (byte)(buf[i + 2] * 0.55 + 255 * 0.45); }
        }
        System.Runtime.InteropServices.Marshal.Copy(buf, 0, d.Scan0, buf.Length);
        outBmp.UnlockBits(d);
        m.Dispose();
        return outBmp;
    }
}
"@

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$font = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Bold)
$small = New-Object System.Drawing.Font("Segoe UI", 10)
$labelW = 190; $headH = 30; $gap = 6

foreach ($cat in $Categories) {
    $types = Get-ChildItem "$Root\$cat\test" -Directory | Sort-Object { $_.Name -ne "good" }, Name
    $cols = $PerType * 2
    $W = $labelW + $cols * ($Cell + $gap)
    $H = $headH + $types.Count * ($Cell + $gap)
    $canvas = New-Object System.Drawing.Bitmap($W, $H)
    $g = [System.Drawing.Graphics]::FromImage($canvas)
    $g.Clear([System.Drawing.Color]::White)
    $g.InterpolationMode = "HighQualityBicubic"
    for ($k = 0; $k -lt $PerType; $k++) {
        $x = $labelW + $k * 2 * ($Cell + $gap)
        $g.DrawString("Mẫu $($k+1): ảnh gốc", $small, [System.Drawing.Brushes]::Black, $x, 6)
        $g.DrawString("Mẫu $($k+1): vùng lỗi (đỏ)", $small, [System.Drawing.Brushes]::Black, $x + $Cell + $gap, 6)
    }
    $row = 0
    foreach ($t in $types) {
        $y = $headH + $row * ($Cell + $gap)
        $n = (Get-ChildItem $t.FullName -File).Count
        $g.DrawString($t.Name, $font, [System.Drawing.Brushes]::Black, 8, $y + $Cell / 2 - 22)
        $g.DrawString("$n ảnh test", $small, [System.Drawing.Brushes]::DimGray, 8, $y + $Cell / 2 + 4)
        $files = Get-ChildItem $t.FullName -File | Sort-Object Name | Select-Object -First $PerType
        $k = 0
        foreach ($f in $files) {
            $x = $labelW + $k * 2 * ($Cell + $gap)
            $img = New-Object System.Drawing.Bitmap($f.FullName)
            $g.DrawImage($img, $x, $y, $Cell, $Cell)
            $maskPath = "$Root\$cat\ground_truth\$($t.Name)\$($f.BaseName)_mask.png"
            if (Test-Path $maskPath) {
                $mask = New-Object System.Drawing.Bitmap($maskPath)
                $ov = [Overlay]::Apply($img, $mask)
                $g.DrawImage($ov, $x + $Cell + $gap, $y, $Cell, $Cell)
                $ov.Dispose(); $mask.Dispose()
            } else {
                $g.DrawImage($img, $x + $Cell + $gap, $y, $Cell, $Cell)
                $g.DrawString("không có lỗi", $small, [System.Drawing.Brushes]::Green, $x + $Cell + $gap + 6, $y + 6)
            }
            $img.Dispose()
            $k++
        }
        $row++
    }
    $out = Join-Path $OutDir "$cat`_defects.png"
    $canvas.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $canvas.Dispose()
    Write-Output $out
}
