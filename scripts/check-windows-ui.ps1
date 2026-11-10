param(
    [string]$Application = 'dist\Observatory.exe',
    [string]$OutputDirectory = 'output\windows-ui'
)

$ErrorActionPreference = 'Stop'
# Never send UI input on a developer's machine, including self-hosted runners.
if ($env:GITHUB_ACTIONS -ne 'true' -or $env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    throw 'Native UI checks run only on GitHub-hosted remote runners.'
}

Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Drawing, System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class RemoteWindow {
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd, out Rect rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr hwnd, int x, int y, int width, int height, bool repaint);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
}
'@

$applicationPath = (Resolve-Path -LiteralPath $Application).Path
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$outputPath = (Resolve-Path -LiteralPath $OutputDirectory).Path
$script:appWindow = $null
$script:windowHandle = [IntPtr]::Zero
$script:checks = New-Object 'System.Collections.Generic.List[string]'
$script:captures = New-Object 'System.Collections.Generic.List[string]'

function Wait-Result([scriptblock]$Query, [string]$Description, [int]$Seconds = 20) {
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    do {
        try {
            $value = & $Query
            if ($null -ne $value -and $value -ne $false) { return $value }
        } catch {
            # Providers can replace elements during page and tab transitions.
        }
        Start-Sleep -Milliseconds 150
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for $Description."
}

function Find-Control([string]$Name, [System.Windows.Automation.ControlType]$Type = $null) {
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $Name)
    if ($null -ne $Type) {
        $typeCondition = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty, $Type)
        $condition = New-Object System.Windows.Automation.AndCondition($condition, $typeCondition)
    }
    return $script:appWindow.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
}

function Wait-Control([string]$Name, [System.Windows.Automation.ControlType]$Type = $null) {
    return Wait-Result { Find-Control $Name $Type } "control '$Name'"
}

function Invoke-Button([string]$Name) {
    $control = Wait-Control $Name ([System.Windows.Automation.ControlType]::Button)
    if (-not $control.Current.IsEnabled) { throw "Button '$Name' is disabled." }
    $pattern = $control.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
    $pattern.Invoke()
}

function Select-Tab([string]$Name) {
    $control = Wait-Control $Name ([System.Windows.Automation.ControlType]::TabItem)
    Click-Control $control
}

function Click-Control([System.Windows.Automation.AutomationElement]$Control) {
    $bounds = $Control.Current.BoundingRectangle
    if ($bounds.Width -le 0 -or $bounds.Height -le 0) { throw 'Control has no visible native bounds.' }
    [RemoteWindow]::SetForegroundWindow($script:windowHandle) | Out-Null
    [RemoteWindow]::SetCursorPos([int]($bounds.X+$bounds.Width/2), [int]($bounds.Y+$bounds.Height/2)) | Out-Null
    [RemoteWindow]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)
    [RemoteWindow]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 150
}

function Send-AppKeys([string]$Keys) {
    $focused = [System.Windows.Automation.AutomationElement]::FocusedElement
    if ($null -eq $focused -or $focused.Current.ProcessId -ne $appPid) {
        throw 'Refusing keyboard input outside the packaged application.'
    }
    [System.Windows.Forms.SendKeys]::SendWait($Keys)
    Start-Sleep -Milliseconds 150
}

function Read-Texts {
    $elements = $script:appWindow.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition)
    return @($elements | ForEach-Object { $_.Current.Name })
}

function Capture([string]$Name) {
    Start-Sleep -Milliseconds 300
    $rect = New-Object RemoteWindow+Rect
    if (-not [RemoteWindow]::GetWindowRect($script:windowHandle, [ref]$rect)) { throw 'Window bounds unavailable.' }
    $bitmap = New-Object System.Drawing.Bitmap(($rect.Right-$rect.Left), ($rect.Bottom-$rect.Top))
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $dc = $graphics.GetHdc()
    try {
        if (-not [RemoteWindow]::PrintWindow($script:windowHandle, $dc, 0)) { throw 'Window capture failed.' }
    } finally {
        $graphics.ReleaseHdc($dc)
        $graphics.Dispose()
    }
    try {
        $bitmap.Save((Join-Path $outputPath ($Name+'.png')), [System.Drawing.Imaging.ImageFormat]::Png)
    } finally { $bitmap.Dispose() }
    $script:appWindow.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition) |
        ForEach-Object { $_.Current.ControlType.ProgrammaticName+': '+$_.Current.Name } |
        Set-Content -LiteralPath (Join-Path $outputPath ($Name+'.txt')) -Encoding UTF8
    $script:captures.Add($Name+'.png')
}

$env:QT_QPA_PLATFORM = 'windows'
$appProcess = Start-Process -FilePath $applicationPath -PassThru
$appPid = $null
$failure = $null
try {
    $script:appWindow = Wait-Result {
        $condition = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::NameProperty, 'Observatory')
        $candidates = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Children, $condition)
        foreach ($candidate in $candidates) {
            $process = Get-Process -Id $candidate.Current.ProcessId -ErrorAction SilentlyContinue
            if ($process -and $process.Path -eq $applicationPath) { return $candidate }
        }
    } 'the packaged application window' 40
    $appPid = $script:appWindow.Current.ProcessId
    $script:windowHandle = [IntPtr]$script:appWindow.Current.NativeWindowHandle
    [RemoteWindow]::MoveWindow($script:windowHandle, 0, 0, 1280, 900, $true) | Out-Null
    [RemoteWindow]::SetForegroundWindow($script:windowHandle) | Out-Null

    Wait-Control 'Choose experiment' | Out-Null
    Wait-Control 'Load saved experiment' | Out-Null
    Capture '01-starting-menu'
    $script:checks.Add('Packaged application opens at the starting menu')

    Invoke-Button 'Choose experiment'
    Wait-Control 'Open frozen starting setup' | Out-Null
    Capture '02-experiment-library'
    Invoke-Button 'Open frozen starting setup'
    Wait-Control 'Run' | Out-Null
    $name = Wait-Control 'Experiment name'
    if ($name.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern).Current.Value -ne 'Solar System') {
        throw 'The library did not select the Solar System starting setup.'
    }
    Wait-Control 'Sun' | Out-Null
    Wait-Result { (Read-Texts | Where-Object { $_ -like 'Starting setup*' }).Count -gt 0 } 'the frozen starting setup' | Out-Null
    Capture '03-frozen-solar-system'
    $script:checks.Add('Library opens the frozen Solar System')

    Invoke-Button 'Run'
    Wait-Control 'Pause' | Out-Null
    Start-Sleep -Milliseconds 500
    Invoke-Button 'Pause'
    Capture '04-paused-run'
    Invoke-Button 'Reset'
    $script:checks.Add('Run, pause and return to starting setup')

    # Qt virtualizes list rows; use the visible item rather than counting UIA children.
    $sun = Wait-Control 'Sun' ([System.Windows.Automation.ControlType]::ListItem)
    Click-Control $sun
    Select-Tab 'Edit body'
    $beforeMass = (Read-Texts | Where-Object { $_ -like 'Resulting mass:*' } | Select-Object -First 1)
    $mass = Wait-Control 'Mass multiplier' ([System.Windows.Automation.ControlType]::Spinner)
    Click-Control $mass
    Send-AppKeys '^a2{TAB}'
    Invoke-Button 'Apply to starting setup'
    Wait-Result { (Read-Texts | Where-Object { $_ -like 'Resulting mass:*' -and $_ -ne $beforeMass }).Count -gt 0 } 'updated stellar mass' | Out-Null
    Capture '05-edited-stellar-mass'
    $script:checks.Add('Edit the starting stellar mass through the native control')

    Select-Tab 'Add body'
    $bodyType = Wait-Control 'Body type' ([System.Windows.Automation.ControlType]::ComboBox)
    Click-Control $bodyType
    Send-AppKeys '{END}{ENTER}'
    Wait-Control 'Vesta' ([System.Windows.Automation.ControlType]::Text) | Out-Null
    Wait-Control 'Body definition' | Out-Null
    Capture '06-body-catalog'
    Invoke-Button 'Add'
    Wait-Result { (Read-Texts | Where-Object { $_ -like 'Click in the scene to place*' }).Count -gt 0 } 'scene placement prompt' | Out-Null
    $scene = Wait-Control 'Simulation scene'
    $bounds = $scene.Current.BoundingRectangle
    if ($bounds.Width -lt 100 -or $bounds.Height -lt 100) { throw 'Scene has no usable native bounds.' }
    [RemoteWindow]::SetForegroundWindow($script:windowHandle) | Out-Null
    [RemoteWindow]::SetCursorPos([int]($bounds.X+$bounds.Width*.8), [int]($bounds.Y+$bounds.Height*.25)) | Out-Null
    [RemoteWindow]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)
    [RemoteWindow]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)
    Wait-Result { (Read-Texts | Where-Object { $_ -like 'Body added to the starting setup.*' }).Count -gt 0 } 'successful scene placement' | Out-Null
    Capture '07-body-placed'
    Invoke-Button 'Undo edit'
    Wait-Result { (Read-Texts | Where-Object { $_ -like 'Restored the previous starting setup.*' }).Count -gt 0 } 'undo body placement' | Out-Null
    Invoke-Button 'Restore original'
    Capture '08-original-restored'
    $script:checks.Add('Catalog, scene placement, undo and restore original')
} catch {
    $failure = $_.ToString()
    if ($null -ne $script:appWindow) {
        try { Capture 'failure' } catch { Write-Warning $_ }
    }
} finally {
    if ($appPid) { Stop-Process -Id $appPid -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $appProcess.Id -Force -ErrorAction SilentlyContinue
    @{ success=($null -eq $failure); checks=@($script:checks); screenshots=@($script:captures); error=$failure } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputPath 'report.json') -Encoding UTF8
}
if ($failure) { throw $failure }
Write-Output ($script:checks -join "`n")
