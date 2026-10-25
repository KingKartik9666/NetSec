# A few changes in NetSec 0.4.0

I’ve kept the current look and spent this update improving the app underneath it.

- You can now choose from ten models, compare a few together, or run all compatible models in extensive mode.
- The desktop stays on Electron, with a new React and TypeScript interface and a Rust backend. Model inference runs natively on your CPU.
- Results arrive separately, so one model failing doesn’t wipe out the others. You can cancel the remaining work and keep completed results.
- Explanations and the local English assistant are available alongside each result. I also fixed a live-screen issue that could show an old result after stopping monitoring.
- Startup, window controls and shutdown have been checked in the actual desktop app. The existing theme is still here.

The Windows setup updates an existing installation. A full app-folder ZIP is also provided for people who prefer to extract the app themselves. Packet capture still needs the separate Npcap driver; samples and CSV analysis work locally without it.

I’ve also added a file audit explaining what the project still needs, what is only useful for development or rollback, and which old build files can be cleaned up later.

This is a local Windows build. It hasn’t been signed or certified for the Microsoft Store yet.
