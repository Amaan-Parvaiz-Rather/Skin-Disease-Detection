/* ==========================================================================
   DermānAI — app.js
   Handles: Nav drawer | Modal | Upload | Scan | Results | Counters
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {

    /* ─────────────────────────────────────────────────────────────────
       1. STAGGERED MENU (GSAP)
    ───────────────────────────────────────────────────────────────── */
    const smWrapper = document.querySelector('.staggered-menu-wrapper');
    const smToggle = document.getElementById('sm-toggle-btn');
    const smPanel = document.getElementById('staggered-menu-panel');
    const smPreLayers = document.getElementById('sm-prelayers');
    const smIcon = document.getElementById('sm-icon');
    const smPlusH = document.getElementById('sm-plus-h');
    const smPlusV = document.getElementById('sm-plus-v');

    let smOpen = false;
    let smBusy = false;
    let smOpenTl = null;

    // Set initial GSAP states
    if (typeof gsap !== 'undefined' && smPanel) {
        const preLayers = Array.from(smPreLayers.querySelectorAll('.sm-prelayer'));
        const offscreen = -100; // Left side
        
        gsap.set([smPanel, ...preLayers], { xPercent: offscreen, opacity: 1 });
        gsap.set(smPreLayers, { xPercent: 0, opacity: 1 });
        
        // Initial state for hamburger (two horizontal lines)
        gsap.set(smPlusH, { y: -3, rotate: 0 });
        gsap.set(smPlusV, { y: 3, rotate: 0 });

        function buildOpenTimeline() {
            const itemEls = Array.from(smPanel.querySelectorAll('.sm-panel-itemLabel'));
            const numberEls = Array.from(smPanel.querySelectorAll('.sm-panel-list[data-numbering] .sm-panel-item'));
            const socialTitle = smPanel.querySelector('.sm-socials-title');
            const socialLinks = Array.from(smPanel.querySelectorAll('.sm-socials-link'));

            if (itemEls.length) gsap.set(itemEls, { yPercent: 140, rotate: 10 });
            if (numberEls.length) gsap.set(numberEls, { '--sm-num-opacity': 0 });
            if (socialTitle) gsap.set(socialTitle, { opacity: 0 });
            if (socialLinks.length) gsap.set(socialLinks, { y: 25, opacity: 0 });

            const tl = gsap.timeline({ paused: true, onComplete: () => smBusy = false });

            // Animate prelayers
            preLayers.forEach((el, i) => {
                tl.fromTo(el, { xPercent: offscreen }, { xPercent: 0, duration: 0.5, ease: 'power4.out' }, i * 0.07);
            });

            const panelInsertTime = preLayers.length ? (preLayers.length - 1) * 0.07 + 0.08 : 0;
            const panelDuration = 0.65;

            // Animate panel
            tl.fromTo(smPanel, { xPercent: offscreen }, { xPercent: 0, duration: panelDuration, ease: 'power4.out' }, panelInsertTime);

            // Animate items
            if (itemEls.length) {
                const itemsStart = panelInsertTime + panelDuration * 0.15;
                tl.to(itemEls, { yPercent: 0, rotate: 0, duration: 1, ease: 'power4.out', stagger: { each: 0.1, from: 'start' } }, itemsStart);
                if (numberEls.length) {
                    tl.to(numberEls, { duration: 0.6, ease: 'power2.out', '--sm-num-opacity': 1, stagger: { each: 0.08, from: 'start' } }, itemsStart + 0.1);
                }
            }

            // Animate socials
            if (socialTitle || socialLinks.length) {
                const socialsStart = panelInsertTime + panelDuration * 0.4;
                if (socialTitle) tl.to(socialTitle, { opacity: 1, duration: 0.5, ease: 'power2.out' }, socialsStart);
                if (socialLinks.length) {
                    tl.to(socialLinks, { y: 0, opacity: 1, duration: 0.55, ease: 'power3.out', stagger: { each: 0.08, from: 'start' }, onComplete: () => gsap.set(socialLinks, { clearProps: 'opacity' }) }, socialsStart + 0.04);
                }
            }

            return tl;
        }

        function playClose() {
            if (smOpenTl) { smOpenTl.kill(); smOpenTl = null; }
            const all = [...preLayers, smPanel];
            gsap.to(all, {
                xPercent: -100, duration: 0.32, ease: 'power3.in', overwrite: 'auto',
                onComplete: () => {
                    smBusy = false;
                    smWrapper.removeAttribute('data-open');
                }
            });
        }

        function animateIcon(opening) {
            if (opening) {
                gsap.to(smPlusH, { y: 0, rotate: 45, duration: 0.4, ease: 'power3.out', overwrite: 'auto' });
                gsap.to(smPlusV, { y: 0, rotate: -45, duration: 0.4, ease: 'power3.out', overwrite: 'auto' });
            } else {
                gsap.to(smPlusH, { y: -3, rotate: 0, duration: 0.35, ease: 'power3.inOut', overwrite: 'auto' });
                gsap.to(smPlusV, { y: 3, rotate: 0, duration: 0.35, ease: 'power3.inOut', overwrite: 'auto' });
            }
        }

        function toggleMenu() {
            if (smBusy) return;
            smBusy = true;
            smOpen = !smOpen;
            
            if (smOpen) {
                smWrapper.setAttribute('data-open', 'true');
                smOpenTl = buildOpenTimeline();
                smOpenTl.play(0);
            } else {
                playClose();
            }
            animateIcon(smOpen);
        }

        smToggle.addEventListener('click', toggleMenu);

        // Close when clicking internal links
        document.querySelectorAll('.drawer-link').forEach(link => {
            link.addEventListener('click', () => {
                if (smOpen) toggleMenu();
            });
        });

        // The "Scan Now" button in the menu
        const menuScanBtn = document.getElementById('menu-scan-btn');
        if (menuScanBtn) {
            menuScanBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (smOpen) toggleMenu();
                // We rely on the global openScanModal function
                if (typeof openScanModal === 'function') openScanModal();
            });
        }

        // Close on click away
        document.addEventListener('mousedown', (e) => {
            if (smOpen && !smPanel.contains(e.target) && !smToggle.contains(e.target)) {
                toggleMenu();
            }
        });
    }


    /* ─────────────────────────────────────────────────────────────────
       3. STAT COUNTER ANIMATION (triggered when stats bar is visible)
    ───────────────────────────────────────────────────────────────── */
    function animateCounter(el, target, duration = 1600) {
        let start = 0;
        const step = (timestamp) => {
            if (!start) start = timestamp;
            const progress = Math.min((timestamp - start) / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.floor(eased * target);
            if (progress < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }

    const statsBar = document.querySelector('.stats-bar');
    if (statsBar) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    document.querySelectorAll('.stat-number').forEach(el => {
                        const target = parseInt(el.dataset.target, 10);
                        animateCounter(el, target);
                    });
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });
        observer.observe(statsBar);
    }


    /* ─────────────────────────────────────────────────────────────────
       4. SCAN MODAL — open / close + camera logic
    ───────────────────────────────────────────────────────────────── */
    const scanModal   = document.getElementById('scan-modal');
    const mainView    = document.getElementById('main-view');
    const resultsSection = document.getElementById('results-section');

    // Camera state
    let camStream      = null;
    let camFacingMode  = 'environment'; // prefer back camera

    const camScreen    = document.getElementById('cam-screen');
    const permScreen   = document.getElementById('perm-screen');
    const uploadScreen = document.getElementById('upload-screen');

    function showScreen(name) {
        camScreen.style.display    = 'none';
        permScreen.style.display   = 'none';
        uploadScreen.style.display = 'none';
        if (name === 'cam')    { camScreen.style.display    = 'flex'; }
        if (name === 'perm')   { permScreen.style.display   = 'flex'; }
        if (name === 'upload') { uploadScreen.style.display = 'block'; }
    }

    async function startCamera(facingMode) {
        // Stop any existing stream
        if (camStream) camStream.getTracks().forEach(t => t.stop());
        try {
            camStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false
            });
            const video = document.getElementById('cam-video');
            video.srcObject = camStream;
            showScreen('cam');
        } catch (err) {
            // Permission denied or no camera
            stopCamera();
            showScreen('upload');
        }
    }

    function stopCamera() {
        if (camStream) {
            camStream.getTracks().forEach(t => t.stop());
            camStream = null;
        }
    }

    async function tryOpenCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            // No camera API — go straight to upload
            showScreen('upload');
            return;
        }
        try {
            // Check permission state without requesting yet (if possible)
            if (navigator.permissions) {
                const status = await navigator.permissions.query({ name: 'camera' });
                if (status.state === 'denied') {
                    showScreen('upload');
                    return;
                }
                if (status.state === 'prompt') {
                    showScreen('perm');
                    return;
                }
            }
            // Permission is 'granted' — go straight to camera
            await startCamera(camFacingMode);
        } catch (e) {
            showScreen('perm');
        }
    }

    function openScanModal() {
        scanModal.classList.add('active');
        document.body.style.overflow = 'hidden';
        tryOpenCamera();
    }

    function closeScanModal() {
        scanModal.classList.remove('active');
        document.body.style.overflow = '';
        stopCamera();
        resetUpload();
    }

    // Permission screen buttons
    document.getElementById('perm-allow-btn').addEventListener('click', async () => {
        await startCamera(camFacingMode);
    });

    document.getElementById('perm-skip-btn').addEventListener('click', () => {
        showScreen('upload');
    });

    // In-camera: switch to upload
    document.getElementById('cam-switch-upload').addEventListener('click', () => {
        stopCamera();
        showScreen('upload');
        // Show "Use Camera" button in upload mode
        const camBtn = document.getElementById('upload-use-camera-btn');
        if (camBtn) camBtn.style.display = 'block';
    });

    // In-upload: return to camera
    const uploadUseCameraBtn = document.getElementById('upload-use-camera-btn');
    if (uploadUseCameraBtn) {
        uploadUseCameraBtn.addEventListener('click', async () => {
            uploadUseCameraBtn.style.display = 'none';
            await tryOpenCamera();
        });
    }

    // Modal tabs logic
    const tabCamera = document.getElementById('tab-camera');
    if (tabCamera) {
        tabCamera.addEventListener('click', async () => {
            await tryOpenCamera();
        });
    }

    // Flip camera
    document.getElementById('cam-flip-btn').addEventListener('click', async () => {
        camFacingMode = camFacingMode === 'environment' ? 'user' : 'environment';
        await startCamera(camFacingMode);
    });

    // Capture photo from camera
    document.getElementById('cam-capture-btn').addEventListener('click', () => {
        const video  = document.getElementById('cam-video');
        const canvas = document.getElementById('cam-canvas');
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        canvas.toBlob((blob) => {
            const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
            stopCamera();
            showScreen('upload');
            handleFile(file); // Use existing upload pipeline
        }, 'image/jpeg', 0.92);
    });

    // All buttons that open the modal
    const openModalTriggers = [
        document.getElementById('open-scan-modal'),
        document.getElementById('process-scan-btn'),
        document.getElementById('cta-scan-btn'),
        document.getElementById('drawer-scan-btn'), // old drawer btn
        document.getElementById('dock-scan-btn'),
    ];
    openModalTriggers.forEach(btn => {
        if (btn) btn.addEventListener('click', (e) => {
            // Prevent default just in case it's an anchor (even though buttons don't strictly need it, it's safe)
            e.preventDefault();
            // We can't access toggleMenu here, but the menu is probably not open if these buttons are clicked
            openScanModal();
        });
    });

    const closeModalBtn = document.getElementById('close-scan-modal');
    if (closeModalBtn) closeModalBtn.addEventListener('click', closeScanModal);

    // Close on backdrop click
    scanModal.addEventListener('click', (e) => {
        if (e.target === scanModal) closeScanModal();
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (scanModal.classList.contains('active')) closeScanModal();
            if (typeof toggleMenu === 'function' && smOpen) toggleMenu();
        }
    });


    /* ─────────────────────────────────────────────────────────────────
       5. FILE UPLOAD LOGIC
    ───────────────────────────────────────────────────────────────── */
    const uploadArea              = document.getElementById('upload-area');
    const fileInput               = document.getElementById('file-input');
    const uploadPrompt            = document.getElementById('upload-prompt');
    const imagePreviewContainer   = document.getElementById('image-preview-container');
    const imagePreview            = document.getElementById('image-preview');
    const scannerOverlay          = document.getElementById('scanner-overlay');
    const scannerFrame            = document.getElementById('scanner-frame');
    const analyzeBtn              = document.getElementById('analyze-btn');
    const uploadForm              = document.getElementById('upload-form');

    let selectedFile = null;

    // Click on drop zone OR choose-file button → open file dialog
    uploadArea.addEventListener('click', () => fileInput.click());

    const chooseFileBtn = document.getElementById('choose-file-btn');
    if (chooseFileBtn) {
        chooseFileBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // prevent bubbling to uploadArea
            fileInput.click();
        });
    }

    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.style.display = 'none';
            imagePreviewContainer.style.display = 'block';
            analyzeBtn.style.display = 'block';

            uploadArea.style.padding = '0.5rem';
            uploadArea.style.borderStyle = 'solid';
            uploadArea.style.background = 'var(--color-100)';
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        uploadPrompt.style.display = 'flex';
        imagePreviewContainer.style.display = 'none';
        analyzeBtn.style.display = 'none';
        imagePreview.src = '';

        if (scannerOverlay) scannerOverlay.classList.remove('scanning');
        if (scannerFrame) scannerFrame.classList.remove('scanning');

        uploadArea.style.padding = '3rem 1.5rem';
        uploadArea.style.borderStyle = 'dashed';
        uploadArea.style.background = 'var(--color-50)';

        if (analyzeBtn) {
            analyzeBtn.textContent = 'Analyse Now →';
            analyzeBtn.disabled = false;
        }
    }


    /* ─────────────────────────────────────────────────────────────────
       6. FORM SUBMIT — send to /predict, transition to results
    ───────────────────────────────────────────────────────────────── */
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!selectedFile) return;

            // Start scanner animations
            if (scannerOverlay) scannerOverlay.classList.add('scanning');
            if (scannerFrame) scannerFrame.classList.add('scanning');
            analyzeBtn.textContent = 'Analysing…';
            analyzeBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (response.ok) {
                    // Keep scanning animation a moment for effect
                    setTimeout(() => {
                        showResults(data);
                    }, 900);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                    resetUpload();
                }
            } catch (error) {
                alert('An error occurred during analysis. Please try again.');
                console.error('Prediction error:', error);
                resetUpload();
            }
        });
    }


    /* ─────────────────────────────────────────────────────────────────
       7. SHOW RESULTS — transition from homepage to results view
    ───────────────────────────────────────────────────────────────── */
    function showResults(data) {
        // Hide modal and homepage, show results
        scanModal.classList.remove('active');
        mainView.style.display = 'none';
        resultsSection.style.display = 'block';
        document.body.style.overflow = '';
        window.scrollTo({ top: 0, behavior: 'smooth' });

        // Reset state — show skeletons
        document.getElementById('results-content').style.display = 'none';
        document.getElementById('real-images').style.display = 'none';
        document.getElementById('slider-wrapper').style.display = 'none';
        document.getElementById('results-skeleton').style.display = 'block';
        document.getElementById('skel-img').style.display = 'block';

        const origImg = document.getElementById('res-original');
        const heatImg = document.getElementById('res-heatmap');
        const isHealthy = data.prediction && data.prediction.healthy;

        if (isHealthy) {
            // Healthy — only load original image, hide heatmap completely
            heatImg.src = '';
            heatImg.style.display = 'none';

            origImg.onload = () => setTimeout(() => populateResults(data), 400);
            origImg.onerror = () => setTimeout(() => populateResults(data), 400);
            origImg.src = '/' + data.original_image.replace(/\\/g, '/');
        } else {
            // Disease detected — preload both images then reveal
            heatImg.style.display = 'block';

            let loadedCount = 0;
            const onImageLoad = () => {
                loadedCount++;
                if (loadedCount === 2) {
                    setTimeout(() => populateResults(data), 400);
                }
            };

            origImg.onload  = onImageLoad;
            heatImg.onload  = onImageLoad;
            origImg.onerror = onImageLoad;
            heatImg.onerror = onImageLoad;

            origImg.src = '/' + data.original_image.replace(/\\/g, '/');
            heatImg.src = '/' + data.heatmap_image.replace(/\\/g, '/');
        }
    }

    function populateResults(data) {
        // Hide skeletons
        document.getElementById('results-skeleton').style.display = 'none';
        document.getElementById('skel-img').style.display = 'none';

        // Show real content
        document.getElementById('results-content').style.display = 'block';
        document.getElementById('real-images').style.display = 'block';

        const p    = data.prediction;
        const info = data.info;

        if (p.healthy) {
            /* ── HEALTHY SKIN PATH ── */
            document.getElementById('healthy-result').style.display = 'block';
            document.getElementById('disease-result').style.display = 'none';
            // Hide heatmap slider — not useful for healthy skin
            document.getElementById('slider-wrapper').style.display = 'none';

            // Populate skincare tips from backend info.precautions
            const tipsList = document.getElementById('healthy-tips-list');
            tipsList.innerHTML = '';
            (info.precautions || []).forEach(tip => {
                const li = document.createElement('li');
                li.textContent = tip;
                tipsList.appendChild(li);
            });

            document.getElementById('healthy-doctor').textContent =
                info.when_to_see_doctor || '';
        } else {
            /* ── DISEASE DETECTED PATH ── */
            document.getElementById('healthy-result').style.display = 'none';
            document.getElementById('disease-result').style.display = 'block';
            document.getElementById('slider-wrapper').style.display = 'block';

            // Rich Severity Panel
            const sevPanel = document.getElementById('severity-panel');
            if (sevPanel) {
                sevPanel.setAttribute('data-level', p.severity_label);
                
                const sevLabel = document.getElementById('res-severity-label');
                if (sevLabel) sevLabel.textContent = p.severity_label;
                
                const sevScore = document.getElementById('res-severity-score');
                if (sevScore) sevScore.textContent = `${p.severity} / 10`;

                // Update Gauge pointer position
                const pointer = document.getElementById('gauge-pointer');
                if (pointer) {
                    pointer.classList.remove('visible');
                    // Map 0-10 severity to 0-100% position on the gauge
                    let leftPercent = Math.min(Math.max((p.severity / 10) * 100, 5), 95); // keep within bounds
                    
                    setTimeout(() => {
                        pointer.style.left = `${leftPercent}%`;
                        pointer.classList.add('visible');
                    }, 100);
                }

                // Update Plain-English meaning
                const meaningText = document.getElementById('severity-meaning-text');
                if (meaningText) {
                    const meanings = {
                        "Low": "This condition is generally mild. Monitor for any changes or worsening symptoms.",
                        "Moderate": "This condition may require a professional evaluation if it persists.",
                        "High": "This condition requires prompt medical attention and careful monitoring.",
                        "Critical": "Urgent medical attention is highly recommended. Please consult a dermatologist."
                    };
                    meaningText.textContent = meanings[p.severity_label] || "Please consult a dermatologist for an accurate assessment.";
                }
            }

            // Disease name (strip dataset suffix)
            document.getElementById('res-disease').textContent =
                p.disease.split(' Photos')[0].split('_').join(' ');

            // Confidence bar (animated)
            document.getElementById('res-confidence-text').textContent = p.confidence + '%';
            setTimeout(() => {
                document.getElementById('res-confidence-bar').style.width = p.confidence + '%';
            }, 100);

            // Info from Gemini
            document.getElementById('res-overview').textContent  = info.overview  || '—';
            document.getElementById('res-symptoms').textContent  = info.symptoms  || '—';
            document.getElementById('res-doctor').textContent    = info.when_to_see_doctor || '—';

            // Precautions list
            const ul = document.getElementById('res-precautions');
            ul.innerHTML = '';
            (info.precautions || []).forEach(prec => {
                const li = document.createElement('li');
                li.textContent = prec;
                ul.appendChild(li);
            });
        }
    }


    /* ─────────────────────────────────────────────────────────────────
       8. HEATMAP BLEND SLIDER
    ───────────────────────────────────────────────────────────────── */
    const blendSlider = document.getElementById('blend-slider');
    const heatmapImg  = document.getElementById('res-heatmap');

    if (blendSlider && heatmapImg) {
        blendSlider.addEventListener('input', (e) => {
            heatmapImg.style.opacity = e.target.value / 100;
        });
    }


    /* ─────────────────────────────────────────────────────────────────
       9. RESET — back to homepage
    ───────────────────────────────────────────────────────────────── */
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            resultsSection.style.display = 'none';
            mainView.style.display = 'block';
            resetUpload();

            // Reset results state
            if (document.getElementById('res-confidence-bar')) {
                document.getElementById('res-confidence-bar').style.width = '0%';
            }
            if (blendSlider) blendSlider.value = 80;
            if (heatmapImg)  heatmapImg.style.opacity = 0.8;

            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    /* ── Scan Again — go back to upload screen without going home ── */
    const scanAgainBtn = document.getElementById('scan-again-btn');
    if (scanAgainBtn) {
        scanAgainBtn.addEventListener('click', () => {
            // Smoothly hide results
            resultsSection.style.display = 'none';
            mainView.style.display = 'block';

            // Reset upload widget
            resetUpload();
            if (blendSlider) blendSlider.value = 80;
            if (heatmapImg)  heatmapImg.style.opacity = 0.8;

            // Open the scan modal right away
            if (scanModal) {
                scanModal.classList.add('active');
                document.body.style.overflow = 'hidden';
                // Jump straight to upload screen (skip permission screen)
                const uploadScreen = document.getElementById('upload-screen');
                const permScreen   = document.getElementById('permission-screen');
                const scanScreen   = document.getElementById('scan-screen');
                if (permScreen)   permScreen.style.display   = 'none';
                if (scanScreen)   scanScreen.style.display   = 'none';
                if (uploadScreen) uploadScreen.style.display = 'block';
            }

            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }



    /* ─────────────────────────────────────────────────────────────────
       10. SCROLL-REVEAL — subtle entrance animation for cards
    ───────────────────────────────────────────────────────────────── */
    const revealTargets = document.querySelectorAll(
        '.process-card, .disease-card, .mockup-item, .stat-pill'
    );

    if ('IntersectionObserver' in window) {
        const revealObs = new IntersectionObserver((entries) => {
            entries.forEach((entry, i) => {
                if (entry.isIntersecting) {
                    entry.target.style.transitionDelay = `${(i % 4) * 60}ms`;
                    entry.target.classList.add('revealed');
                    revealObs.unobserve(entry.target);
                }
            });
        }, { threshold: 0.15 });

        // Add initial hidden state via JS (so CSS is not broken if JS is off)
        revealTargets.forEach(el => {
            el.style.opacity    = '0';
            el.style.transform  = 'translateY(24px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease, box-shadow 0.3s';
            revealObs.observe(el);
        });

        // Inject .revealed style
        const revealStyle = document.createElement('style');
        revealStyle.textContent = '.revealed { opacity: 1 !important; transform: translateY(0) !important; }';
        document.head.appendChild(revealStyle);
    }

}); // end DOMContentLoaded
