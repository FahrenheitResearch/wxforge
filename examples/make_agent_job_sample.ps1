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
    [Array]::Copy((Be32 35000000), 0, $Section, 46, 4)
    [Array]::Copy((Be32 97000000), 0, $Section, 50, 4)
    [Array]::Copy((Be32 34000000), 0, $Section, 55, 4)
    [Array]::Copy((Be32 98000000), 0, $Section, 59, 4)
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

function Build-Section5([UInt32]$NumPoints, [single]$ReferenceValue, [byte]$BitsPerValue) {
    $Section = New-Object byte[] 21
    [Array]::Copy((Be32 21), 0, $Section, 0, 4)
    $Section[4] = 5
    [Array]::Copy((Be32 $NumPoints), 0, $Section, 5, 4)
    [Array]::Copy((Be16 0), 0, $Section, 9, 2)
    [Array]::Copy(([BitConverter]::GetBytes([single]$ReferenceValue)[3..0]), 0, $Section, 11, 4)
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

function Build-Message(
    [byte]$Discipline,
    [byte]$Category,
    [byte]$Number,
    [byte]$LevelType,
    [UInt32]$LevelValue,
    [UInt32]$ForecastTime,
    [single]$ReferenceValue,
    [byte[]]$Payload
) {
    $Section1 = Build-Section1
    $Section3 = Build-Section3
    $Section4 = Build-Section4 $Category $Number $LevelType $LevelValue $ForecastTime
    $Section5 = Build-Section5 4 $ReferenceValue 8
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

function Message-Spec($Label, $Category, $Number, $LevelType, $LevelValue, $ReferenceValue, [byte[]]$Payload) {
    @{
        Label = $Label
        Category = [byte]$Category
        Number = [byte]$Number
        LevelType = [byte]$LevelType
        LevelValue = [UInt32]$LevelValue
        ReferenceValue = [single]$ReferenceValue
        Payload = $Payload
    }
}

$PayloadWarm = [byte[]]@(0, 1, 2, 3)
$PayloadMoist = [byte[]]@(0, 2, 4, 6)
$PayloadWind = [byte[]]@(0, 1, 3, 4)
$PayloadHeight = [byte[]]@(0, 5, 10, 15)

$Specs = @(
    (Message-Spec "TMP:2 m above ground" 0 0 103 2 305.0 $PayloadWarm),
    (Message-Spec "RH:2 m above ground" 1 1 103 2 70.0 $PayloadMoist),
    (Message-Spec "UGRD:10 m above ground" 2 2 103 10 8.0 $PayloadWind),
    (Message-Spec "VGRD:10 m above ground" 2 3 103 10 4.0 $PayloadWind),
    (Message-Spec "PRMSL:surface" 3 1 1 0 101200.0 $PayloadHeight),
    (Message-Spec "PRES:surface" 3 0 1 0 100800.0 $PayloadHeight),
    (Message-Spec "HGT:surface" 3 5 1 0 50.0 $PayloadWind),
    (Message-Spec "TMP:1000 hPa" 0 0 100 100000 303.0 $PayloadWarm),
    (Message-Spec "TMP:925 hPa" 0 0 100 92500 299.0 $PayloadWarm),
    (Message-Spec "TMP:850 hPa" 0 0 100 85000 295.0 $PayloadWarm),
    (Message-Spec "TMP:700 hPa" 0 0 100 70000 283.0 $PayloadWarm),
    (Message-Spec "TMP:500 hPa" 0 0 100 50000 263.0 $PayloadWarm),
    (Message-Spec "TMP:300 hPa" 0 0 100 30000 238.0 $PayloadWarm),
    (Message-Spec "RH:1000 hPa" 1 1 100 100000 90.0 $PayloadMoist),
    (Message-Spec "RH:925 hPa" 1 1 100 92500 85.0 $PayloadMoist),
    (Message-Spec "RH:850 hPa" 1 1 100 85000 75.0 $PayloadMoist),
    (Message-Spec "RH:700 hPa" 1 1 100 70000 50.0 $PayloadMoist),
    (Message-Spec "RH:500 hPa" 1 1 100 50000 35.0 $PayloadMoist),
    (Message-Spec "RH:300 hPa" 1 1 100 30000 15.0 $PayloadMoist),
    (Message-Spec "UGRD:1000 hPa" 2 2 100 100000 10.0 $PayloadWind),
    (Message-Spec "UGRD:925 hPa" 2 2 100 92500 14.0 $PayloadWind),
    (Message-Spec "UGRD:850 hPa" 2 2 100 85000 18.0 $PayloadWind),
    (Message-Spec "UGRD:700 hPa" 2 2 100 70000 24.0 $PayloadWind),
    (Message-Spec "UGRD:500 hPa" 2 2 100 50000 32.0 $PayloadWind),
    (Message-Spec "UGRD:300 hPa" 2 2 100 30000 40.0 $PayloadWind),
    (Message-Spec "VGRD:1000 hPa" 2 3 100 100000 5.0 $PayloadWind),
    (Message-Spec "VGRD:925 hPa" 2 3 100 92500 9.0 $PayloadWind),
    (Message-Spec "VGRD:850 hPa" 2 3 100 85000 13.0 $PayloadWind),
    (Message-Spec "VGRD:700 hPa" 2 3 100 70000 18.0 $PayloadWind),
    (Message-Spec "VGRD:500 hPa" 2 3 100 50000 26.0 $PayloadWind),
    (Message-Spec "VGRD:300 hPa" 2 3 100 30000 34.0 $PayloadWind),
    (Message-Spec "HGT:1000 hPa" 3 5 100 100000 100.0 $PayloadHeight),
    (Message-Spec "HGT:925 hPa" 3 5 100 92500 800.0 $PayloadHeight),
    (Message-Spec "HGT:850 hPa" 3 5 100 85000 1500.0 $PayloadHeight),
    (Message-Spec "HGT:700 hPa" 3 5 100 70000 3100.0 $PayloadHeight),
    (Message-Spec "HGT:500 hPa" 3 5 100 50000 5600.0 $PayloadHeight),
    (Message-Spec "HGT:300 hPa" 3 5 100 30000 9200.0 $PayloadHeight)
)

$Messages = New-Object System.Collections.Generic.List[byte[]]
$IdxLines = New-Object System.Collections.Generic.List[string]
$Offset = 0
$MessageNo = 1

foreach ($Spec in $Specs) {
    $Message = Build-Message 0 $Spec.Category $Spec.Number $Spec.LevelType $Spec.LevelValue 0 $Spec.ReferenceValue $Spec.Payload
    $Messages.Add($Message)
    $IdxLines.Add("$MessageNo`:$Offset`:d=2026031618:$($Spec.Label):anl:")
    $Offset += $Message.Length
    $MessageNo += 1
}

$TotalLength = ($Messages | ForEach-Object { $_.Length } | Measure-Object -Sum).Sum
$Bytes = New-Object byte[] $TotalLength
$Cursor = 0
foreach ($Message in $Messages) {
    [Array]::Copy($Message, 0, $Bytes, $Cursor, $Message.Length)
    $Cursor += $Message.Length
}

$GribPath = Join-Path $PSScriptRoot "agent_job_sample.grib2"
$IdxPath = Join-Path $PSScriptRoot "agent_job_sample.idx"
[IO.File]::WriteAllBytes($GribPath, $Bytes)
[IO.File]::WriteAllLines($IdxPath, $IdxLines)

Write-Host "Wrote $GribPath"
Write-Host "Wrote $IdxPath"
