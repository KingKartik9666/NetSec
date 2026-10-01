; The application installs per-user; only the independent Npcap wizard requests
; driver elevation. No Npcap redistributable is embedded in this product.
!macro customInstall
  IfSilent imdai_npcap_done
  IfFileExists "$WINDIR\System32\Npcap\wpcap.dll" imdai_npcap_done
  MessageBox MB_YESNO|MB_ICONQUESTION "Packet inspection needs the separate Npcap driver. Download the verified official installer and open its setup wizard now? Internet access and Windows administrator approval are required. You may choose No and use samples, CSV analysis and connection monitoring immediately." IDNO imdai_npcap_done
  DetailPrint "Downloading and verifying the official Npcap installer..."
  nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File "$INSTDIR\resources\setup\setup-npcap.ps1" -Install'
  Pop $0
  StrCmp $0 "0" imdai_npcap_done
  MessageBox MB_OK|MB_ICONINFORMATION "Npcap setup did not complete. NetSec is installed and its other features work. Install Npcap later from https://npcap.com/#download, then restart NetSec."
  imdai_npcap_done:
!macroend
