# Determinism Check Report

## ✅ Determinism Check Passed

The training run appears to be deterministic.

## ⚙️ Enforced Settings

- **SHELL:** /bin/bash
- **npm_config_yes:** true
- **COLORTERM:** truecolor
- **PIP_NO_INPUT:** true
- **NVM_INC:** /home/ubuntu/.nvm/versions/node/v22.16.0/include/node
- **DISABLE_AUTO_UPDATE:** true
- **TERM_PROGRAM_VERSION:** 1.4.202508140122-nightly
- **COMPOSER_NO_INTERACTION:** 1
- **HOSTNAME:** cursor
- **NO_AT_BRIDGE:** 1
- **PWD:** /workspace
- **XAUTHORITY:** /tmp/xvfb-run.CR7iTh/Xauthority
- **VSCODE_GIT_ASKPASS_NODE:** /home/ubuntu/.vm-daemon/bin/vm-daemon-cursor-b5299d4577c296ee4b87abd4c7aa7c016674a0f024613436400fb6c0acfd8f3d/Cursor-linux-x64/cursor-nightly
- **HOME:** /home/ubuntu
- **LANG:** en_US.UTF-8
- **CARGO_HOME:** /usr/local/cargo
- **WORKSPACE_ROOT_PATH:** /workspace
- **GIT_ASKPASS:** /home/ubuntu/.vm-daemon/bin/vm-daemon-cursor-b5299d4577c296ee4b87abd4c7aa7c016674a0f024613436400fb6c0acfd8f3d/Cursor-linux-x64/resources/app/extensions/git/dist/askpass.sh
- **CHROME_DESKTOP:** cursor-nightly.desktop
- **NVM_DIR:** /home/ubuntu/.nvm
- **VSCODE_GIT_ASKPASS_EXTRA_ARGS:** 
- **CURSOR_AGENT:** 1
- **TERM:** xterm-256color
- **RUSTUP_HOME:** /usr/local/rustup
- **USER:** ubuntu
- **VSCODE_GIT_IPC_HANDLE:** /tmp/vscode-git-cce7a065fe.sock
- **GIT_DISCOVERY_ACROSS_FILESYSTEM:** 0
- **DISPLAY:** :99
- **SHLVL:** 1
- **NVM_CD_FLAGS:** 
- **PAGER:** sh -c "head -n 10000 | cat"
- **RUST_VERSION:** 1.82.0
- **LC_ALL:** en_US.UTF-8
- **VSCODE_GIT_ASKPASS_MAIN:** /home/ubuntu/.vm-daemon/bin/vm-daemon-cursor-b5299d4577c296ee4b87abd4c7aa7c016674a0f024613436400fb6c0acfd8f3d/Cursor-linux-x64/resources/app/extensions/git/dist/askpass-main.js
- **GDK_BACKEND:** x11
- **PATH:** /home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.local/bin:/home/ubuntu/.nvm/versions/node/v22.16.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
- **ORIGINAL_XDG_CURRENT_DESKTOP:** undefined
- **DBUS_SESSION_BUS_ADDRESS:** disabled:
- **NVM_BIN:** /home/ubuntu/.nvm/versions/node/v22.16.0/bin
- **OLDPWD:** /workspace
- **TERM_PROGRAM:** vscode
- **CURSOR_TRACE_ID:** 3c8739536c4b42158335328c4c812446
- **_:** /home/ubuntu/.local/bin/rldk
- **PYTHONPATH:** :/workspace

## 🔧 Recommended Fixes

- Set torch.backends.cudnn.deterministic = True
- Set environment variable CUDA_LAUNCH_BLOCKING=1
- Set torch.backends.cudnn.benchmark = False
- Set environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8
- Use torch.use_deterministic_algorithms(True)

## 📁 Report Location

Full report saved to: `determinism_report.md`
