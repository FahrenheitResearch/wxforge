function Be16([UInt16]$Value) {
    [byte[]]@((($Value -shr 8) -band 0xFF), ($Value -band 0xFF))
}

function Be32([UInt32]$Value) {
    [byte[]]@(
        (($Value -shr 24) -band 0xFF),
        (($Value -shr 16) -band 0xFF),
        (($Value -shr 8) -band 0xFF),
        ($Value -band 0xFF)
    )
}

function Be64([UInt64]$Value) {
    [byte[]]@(
        (($Value -shr 56) -band 0xFF),
        (($Value -shr 48) -band 0xFF),
        (($Value -shr 40) -band 0xFF),
        (($Value -shr 32) -band 0xFF),
        (($Value -shr 24) -band 0xFF),
        (($Value -shr 16) -band 0xFF),
        (($Value -shr 8) -band 0xFF),
        ($Value -band 0xFF)
    )
}

function Build-Section1 {
    $Section = New-Object byte[] 21
    [Array]::Copy((Be32 21), 0, $Section, 0, 4)
    $Section[4] = 1
    [Array]::Copy((Be16 7), 0, $Section, 5, 2)
    [Array]::Copy((Be16 0), 0, $Section, 7, 2)
    $Section[9] = 28
    $Section[11] = 1
    [Array]::Copy((Be16 2026), 0, $Section, 12, 2)
    $Section[14] = 3
    $Section[15] = 16
    $Section[16] = 18
    $Section[20] = 1
    $Section
}

function Build-Section3 {
    $Section = New-Object byte[] 72
    [Array]::Copy((Be32 72), 0, $Section, 0, 4)
    $Section[4] = 3
    [Array]::Copy((Be32 4), 0, $Section, 6, 4)
    $Section[14] = 6
    [Array]::Copy((Be32 2), 0, $Section, 30, 4)
    [Array]::Copy((Be32 2), 0, $Section, 34, 4)
    [Array]::Copy((Be32 41000000), 0, $Section, 46, 4)
    [Array]::Copy((Be32 100000000), 0, $Section, 50, 4)
    [Array]::Copy((Be32 40000000), 0, $Section, 55, 4)
    [Array]::Copy((Be32 101000000), 0, $Section, 59, 4)
    [Array]::Copy((Be32 1000000), 0, $Section, 63, 4)
    [Array]::Copy((Be32 1000000), 0, $Section, 67, 4)
    $Section
}

function Build-Section4([byte]$Category, [byte]$Number, [byte]$LevelType, [UInt32]$LevelValue, [UInt32]$ForecastTime) {
    $Section = New-Object byte[] 34
    [Array]::Copy((Be32 34), 0, $Section, 0, 4)
    $Section[4] = 4
    $Section[9] = $Category
    $Section[10] = $Number
    $Section[11] = 2
    $Section[17] = 1
    [Array]::Copy((Be32 $ForecastTime), 0, $Section, 18, 4)
    $Section[22] = $LevelType
    [Array]::Copy((Be32 $LevelValue), 0, $Section, 24, 4)
    $Section
}

function Build-Section5([UInt16]$Template, [UInt32]$NumPoints, [single]$ReferenceValue, [Int16]$BinaryScale, [Int16]$DecimalScale, [byte]$BitsPerValue) {
    $Section = New-Object byte[] 21
    [Array]::Copy((Be32 21), 0, $Section, 0, 4)
    $Section[4] = 5
    [Array]::Copy((Be32 $NumPoints), 0, $Section, 5, 4)
    [Array]::Copy((Be16 $Template), 0, $Section, 9, 2)
    [Array]::Copy(([BitConverter]::GetBytes([single]$ReferenceValue)[3..0]), 0, $Section, 11, 4)
    [Array]::Copy(([BitConverter]::GetBytes([Int16]$BinaryScale)[1..0]), 0, $Section, 15, 2)
    [Array]::Copy(([BitConverter]::GetBytes([Int16]$DecimalScale)[1..0]), 0, $Section, 17, 2)
    $Section[19] = $BitsPerValue
    $Section
}

function Build-Section7([byte[]]$Payload) {
    $Section = New-Object byte[] (5 + $Payload.Length)
    [Array]::Copy((Be32 (5 + $Payload.Length)), 0, $Section, 0, 4)
    $Section[4] = 7
    [Array]::Copy($Payload, 0, $Section, 5, $Payload.Length)
    $Section
}

function Build-Message([byte]$Discipline, [byte]$Category, [byte]$Number, [byte]$LevelType, [UInt32]$LevelValue, [UInt32]$ForecastTime, [single]$ReferenceValue, [byte]$BitsPerValue, [byte[]]$Payload) {
    $Section1 = Build-Section1
    $Section3 = Build-Section3
    $Section4 = Build-Section4 $Category $Number $LevelType $LevelValue $ForecastTime
    $Template = if ($BitsPerValue -eq 32 -or $BitsPerValue -eq 64) { 4 } else { 0 }
    $Section5 = Build-Section5 $Template 4 $ReferenceValue 0 0 $BitsPerValue
    $Section6 = [byte[]]@(0, 0, 0, 6, 6, 255)
    $Section7 = Build-Section7 $Payload
    $Marker = [System.Text.Encoding]::ASCII.GetBytes("7777")
    $TotalLength = [UInt64](16 + $Section1.Length + $Section3.Length + $Section4.Length + $Section5.Length + $Section6.Length + $Section7.Length + $Marker.Length)

    $Message = New-Object byte[] $TotalLength
    [Array]::Copy([System.Text.Encoding]::ASCII.GetBytes("GRIB"), 0, $Message, 0, 4)
    $Message[6] = $Discipline
    $Message[7] = 2
    [Array]::Copy((Be64 $TotalLength), 0, $Message, 8, 8)

    $Cursor = 16
    foreach ($Section in @($Section1, $Section3, $Section4, $Section5, $Section6, $Section7, $Marker)) {
        [Array]::Copy($Section, 0, $Message, $Cursor, $Section.Length)
        $Cursor += $Section.Length
    }
    $Message
}

$Message1 = Build-Message 0 0 0 103 2 3 250.0 8 ([byte[]]@(1, 2, 3, 4))
$Message2 = Build-Message 0 2 2 103 10 6 0.0 8 ([byte[]]@(5, 6, 7, 8))
$Bytes = New-Object byte[] ($Message1.Length + $Message2.Length)
[Array]::Copy($Message1, 0, $Bytes, 0, $Message1.Length)
[Array]::Copy($Message2, 0, $Bytes, $Message1.Length, $Message2.Length)

[IO.File]::WriteAllBytes((Join-Path $PSScriptRoot "sample.grib2"), $Bytes)
$IdxLines = @(
    "1:0:d=2026031618:TMP:2 m above ground:anl:",
    "2:$($Message1.Length):d=2026031618:UGRD:10 m above ground:anl:"
)
[IO.File]::WriteAllLines((Join-Path $PSScriptRoot "sample.idx"), $IdxLines)

Write-Host "Wrote $PSScriptRoot\\sample.grib2"
Write-Host "Wrote $PSScriptRoot\\sample.idx"
