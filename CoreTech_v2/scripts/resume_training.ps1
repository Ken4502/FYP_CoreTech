# Script to unfreeze / resume suspended CoreTech v2 training processes
$source = @"
using System;
using System.Runtime.InteropServices;
public class ProcessController {
    [DllImport("ntdll.dll", SetLastError = true)]
    public static extern uint NtResumeProcess(IntPtr processHandle);
}
"@
Add-Type -TypeDefinition $source -ErrorAction SilentlyContinue

$pids = (Get-Process -Name python -ErrorAction SilentlyContinue).Id
foreach ($id in $pids) {
    try {
        $p = [System.Diagnostics.Process]::GetProcessById($id)
        [ProcessController]::NtResumeProcess($p.Handle)
    } catch {}
}
Write-Output "Resumed PIDs: $($pids -join ', ')"
