<#
    Rigenera i PDF della documentazione a partire dai sorgenti HTML in docs/.

        powershell -File scripts\build_docs.ps1

    Usa Microsoft Edge in modalita' headless: e' preinstallato su Windows,
    quindi non serve installare wkhtmltopdf, pandoc o LaTeX.

    Due dettagli imparati a mie spese (vedi diario, 12/08):
      - serve un --user-data-dir dedicato, altrimenti un'istanza di Edge
        gia' aperta tiene il profilo bloccato e il PDF non viene prodotto,
        senza alcun messaggio d'errore;
      - i flag inventati fanno fallire il comando in silenzio: se il file
        non compare, il primo sospetto e' un flag sbagliato.
#>

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$docs = Join-Path $root "docs"
$profileDir = Join-Path $env:TEMP "edge-builddocs-profile"

$candidates = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)
$browser = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $browser) { throw "Nessun browser Chromium trovato: impossibile generare i PDF." }
Write-Host "browser: $browser"

# sorgente (in docs/)  ->  PDF prodotto (nella radice del progetto)
$targets = [ordered]@{
    "manuale_tecnico.html"      = "Manuale_tecnico.pdf"
    "diario_di_bordo.html"      = "Diario_di_bordo.pdf"
    "pipeline_progetto_dl.html" = "Pipeline_progetto_DL.pdf"
    "analisi_paper.html"        = "Analisi_dei_paper.pdf"
}

Get-Process msedge, chrome -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $browser } | Stop-Process -Force -ErrorAction SilentlyContinue

$failed = 0
foreach ($src in $targets.Keys) {
    $srcPath = Join-Path $docs $src
    $outPath = Join-Path $root $targets[$src]

    if (-not (Test-Path $srcPath)) {
        Write-Warning "sorgente mancante: $srcPath"
        $failed++
        continue
    }
    if (Test-Path $outPath) { Remove-Item -LiteralPath $outPath -Force }

    $uri = "file:///" + $srcPath.Replace('\', '/')
    & $browser --headless --disable-gpu --no-sandbox `
        --user-data-dir="$profileDir" `
        --run-all-compositor-stages-before-draw `
        --virtual-time-budget=15000 `
        --print-to-pdf="$outPath" $uri | Out-Null

    if (Test-Path $outPath) {
        $kb = [math]::Round((Get-Item $outPath).Length / 1KB)
        $bytes = [System.IO.File]::ReadAllBytes($outPath)
        $text = [System.Text.Encoding]::GetEncoding(28591).GetString($bytes)
        $pages = ([regex]::Matches($text, '/Type\s*/Page(?!s)')).Count
        Write-Host ("  OK   {0,-26} {1,5} KB  {2,3} pagine" -f $targets[$src], $kb, $pages)
    }
    else {
        Write-Warning ("  FAIL {0}" -f $targets[$src])
        $failed++
    }
}

if ($failed -gt 0) { exit 1 }
Write-Host "documentazione rigenerata."
