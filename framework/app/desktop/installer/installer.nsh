; The application installs per-user; only the independent Npcap wizard requests
; driver elevation. No Npcap redistributable is embedded in this product.
!macro customInstall
  ; Language model pack. The 2.4 GB model file cannot sit inside an NSIS installer (about 2 GB limit), so it ships
  ; next to it. When present beside the installer, it is copied to its sealed runtime location; the runtime checks
  ; its SHA-256 before first use. Without it NetSec installs and runs without the language model.
  ; Keep the file name equal to artifacts/llm/model.json "file" (checked by tools/build_native_desktop.py).
  IfFileExists "$EXEDIR\Qwen3-4B-Instruct-2507-Q4_K_M.gguf" 0 imdai_model_done
  DetailPrint "Installing the local language model (2.4 GB)..."
  CreateDirectory "$INSTDIR\resources\runtime\artifacts\llm"
  CopyFiles /SILENT "$EXEDIR\Qwen3-4B-Instruct-2507-Q4_K_M.gguf" "$INSTDIR\resources\runtime\artifacts\llm" 2400000
  IfFileExists "$INSTDIR\resources\runtime\artifacts\llm\Qwen3-4B-Instruct-2507-Q4_K_M.gguf" imdai_model_done 0
  IfSilent imdai_model_done
  MessageBox MB_OK|MB_ICONINFORMATION "The local language model could not be copied. NetSec is installed and its other features work. Copy Qwen3-4B-Instruct-2507-Q4_K_M.gguf into $INSTDIR\resources\runtime\artifacts\llm later."
  imdai_model_done:
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
