(() => {
    const editor = document.getElementById('editor');
    const statusEl = document.getElementById('status');
    const clearBtn = document.getElementById('clearBtn');
    const downloadBtn = document.getElementById('downloadBtn');

    let ws;
    let clientId = null;
    let connected = false;
    let sendTimer = null;

    function setStatus(text, cls) {
        statusEl.textContent = text;
        statusEl.className = 'badge ' + (cls || 'bg-secondary');
    }

    function wsUrl() {
        const loc = window.location;
        const proto = (loc.protocol === 'https:') ? 'wss:' : 'ws:';
        return proto + '//' + loc.host + '/ws';
    }

    function connect() {
        ws = new WebSocket(wsUrl());
        ws.addEventListener('open', () => {
            connected = true; setStatus('已连接', 'bg-success');
        });
        ws.addEventListener('close', () => { connected = false; setStatus('已断开', 'bg-danger'); setTimeout(connect, 1500); });
        ws.addEventListener('error', () => { setStatus('错误', 'bg-warning'); });
        ws.addEventListener('message', (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                if (msg.type === 'init') {
                    clientId = msg.client_id;
                    editor.innerHTML = msg.content || '';
                } else if (msg.type === 'update') {
                    // apply incoming update
                    // replace only when content differs
                    if (editor.innerHTML !== msg.content) {
                        const sel = saveSelection();
                        editor.innerHTML = msg.content || '';
                        restoreSelection(sel);
                    }
                }
            } catch (e) { console.error(e); }
        });
    }

    // simple debounce send
    function scheduleSend() {
        if (!connected) return;
        if (sendTimer) clearTimeout(sendTimer);
        sendTimer = setTimeout(() => {
            const payload = { type: 'update', content: editor.innerHTML };
            ws.send(JSON.stringify(payload));
            sendTimer = null;
        }, 300);
    }

    editor.addEventListener('input', scheduleSend);

    clearBtn.addEventListener('click', () => {
        editor.innerHTML = '';
        scheduleSend();
        editor.focus();
    });

    downloadBtn.addEventListener('click', () => {
        const blob = new Blob([editor.innerText], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'sharewriter.txt'; a.click();
        URL.revokeObjectURL(url);
    });

    // Ctrl/Cmd+S to download
    window.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
            e.preventDefault(); downloadBtn.click();
        }
    });

    // small helpers to save/restore caret when replacing content
    function saveSelection() {
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0) return null;
        const range = sel.getRangeAt(0);
        return range.cloneRange();
    }
    function restoreSelection(saved) {
        if (!saved) return;
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(saved);
    }

    connect();
    // focus editor on load
    window.addEventListener('load', () => { setTimeout(() => editor.focus(), 200); });
})();
