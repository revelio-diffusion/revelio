document.addEventListener('DOMContentLoaded', () => {
    const imageMapping = {
        bottleneck: [
            "assets/oxfordpets/sd15/step25_mid_k32/351.png",
            "assets/oxfordpets/sd15/step25_mid_k32/3953.png",
            "assets/oxfordpets/sd15/step25_mid_k32/6322.png",
            "assets/oxfordpets/sd15/step25_mid_k32/7206.png",
            "assets/oxfordpets/sd15/step25_mid_k32/20162.png",
            "assets/oxfordpets/sd15/step25_mid_k32/21466.png",
            "assets/oxfordpets/sd15/step25_mid_k32/30711.png",
            "assets/oxfordpets/sd15/step25_mid_k32/38704.png"
        ],
        up_ft0: [
            "assets/oxfordpets/sd15/step25_up0_k32/419.png",
            "assets/oxfordpets/sd15/step25_up0_k32/18581.png",
            "assets/oxfordpets/sd15/step25_up0_k32/23378.png",
            "assets/oxfordpets/sd15/step25_up0_k32/23496.png",
            "assets/oxfordpets/sd15/step25_up0_k32/32770.png",
            "assets/oxfordpets/sd15/step25_up0_k32/37490.png",
            "assets/oxfordpets/sd15/step25_up0_k32/39232.png",
            "assets/oxfordpets/sd15/step25_up0_k32/41164.png",
            "assets/oxfordpets/sd15/step25_up0_k32/55603.png",
            "assets/oxfordpets/sd15/step25_up0_k32/57689.png"
        ],
        up_ft1: [
            "assets/oxfordpets/sd15/step25_up1_k32/7333.png",
            "assets/oxfordpets/sd15/step25_up1_k32/8342.png",
            "assets/oxfordpets/sd15/step25_up1_k32/9841.png",
            "assets/oxfordpets/sd15/step25_up1_k32/10466.png",
            "assets/oxfordpets/sd15/step25_up1_k32/29655.png",
            "assets/oxfordpets/sd15/step25_up1_k32/31487.png",
            "assets/oxfordpets/sd15/step25_up1_k32/41566.png",
            "assets/oxfordpets/sd15/step25_up1_k32/44255.png",
            "assets/oxfordpets/sd15/step25_up1_k32/45549.png",
            "assets/oxfordpets/sd15/step25_up1_k32/71889.png",
            "assets/oxfordpets/sd15/step25_up1_k32/78282.png",
            "assets/oxfordpets/sd15/step25_up1_k32/79494.png",

        ],
        up_ft2: [
            "assets/oxfordpets/sd15/step25_up2_k32/3736.png",
            "assets/oxfordpets/sd15/step25_up2_k32/3768.png",
            "assets/oxfordpets/sd15/step25_up2_k32/4192.png",
            "assets/oxfordpets/sd15/step25_up2_k32/5771.png",
            "assets/oxfordpets/sd15/step25_up2_k32/6164.png",
            "assets/oxfordpets/sd15/step25_up2_k32/11124.png",
            "assets/oxfordpets/sd15/step25_up2_k32/13737.png",
            "assets/oxfordpets/sd15/step25_up2_k32/21391.png",
            "assets/oxfordpets/sd15/step25_up2_k32/22997.png",
            "assets/oxfordpets/sd15/step25_up2_k32/30284.png",
            "assets/oxfordpets/sd15/step25_up2_k32/30507.png",
            "assets/oxfordpets/sd15/step25_up2_k32/32226.png",
            "assets/oxfordpets/sd15/step25_up2_k32/33804.png",
            "assets/oxfordpets/sd15/step25_up2_k32/36898.png",
            "assets/oxfordpets/sd15/step25_up2_k32/39961.png",
            "assets/oxfordpets/sd15/step25_up2_k32/40355.png",

        ]
    };

    const gridItems = document.querySelectorAll('.grid-item');

    gridItems.forEach(item => {
        const info = item.getAttribute('data-info');
        const previews = item.getAttribute('data-preview').split(',');

        item.addEventListener('mouseenter', () => {
            const previewImages = document.createElement('div');
            previewImages.classList.add('preview-images');

            previews.slice(0, 8).forEach(src => { // Use up to 8 images
                const img = document.createElement('img');
                img.src = src;
                img.alt = 'Preview Image';
                previewImages.appendChild(img);
            });

            item.appendChild(previewImages);
        });

        item.addEventListener('mouseleave', () => {
            const previewImages = item.querySelector('.preview-images');
            if (previewImages) {
                previewImages.remove();
            }
        });

        item.addEventListener('click', function () {
            const modal = document.getElementById('featureModal');
            const modalTitle = document.getElementById('featureModalTitle');
            const modalDescription = document.getElementById('featureModalDescription');
            const modalImages = document.getElementById('featureModalImages');

            modalTitle.innerText = info.charAt(0).toUpperCase() + info.slice(1);
            modalDescription.innerText = `Top Activating Images at "${info}"`;

            modalImages.innerHTML = '';

            const images = imageMapping[info];
            if (images && images.length > 0) {
                images.forEach((src, index) => {
                    const imgElement = document.createElement('img');
                    imgElement.src = src;
                    imgElement.alt = info;
                    imgElement.loading = 'lazy';
                    imgElement.classList.add('modal-image'); // Add a class for styling
                
                    imgElement.style.backgroundColor = '#ffe5b4'; // Light peach for odd-indexed images
                    
                
                    modalImages.appendChild(imgElement);
                });
                
            } else {
                modalImages.innerText = 'No images available for this category.';
            }

            modal.style.display = "block";
        });
    });

    document.getElementById("closeFeatureModal").onclick = function () {
        document.getElementById("featureModal").style.display = "none";
    };

    window.addEventListener('click', function (event) {
        const featureModal = document.getElementById("featureModal");
        if (event.target === featureModal) {
            featureModal.style.display = "none";
        }
    });

    document.getElementById("heroImage").onclick = function () {
        document.getElementById("imageModal").style.display = "block";
    };

    document.getElementById("closeModal").onclick = function () {
        document.getElementById("imageModal").style.display = "none";
    };

    window.addEventListener('click', function (event) {
        const imageModal = document.getElementById("imageModal");
        if (event.target === imageModal) {
            imageModal.style.display = "none";
        }
    });

    const menuToggle = document.querySelector('.menu-toggle');
    const nav = document.querySelector('nav');

    if (menuToggle && nav) {
        menuToggle.addEventListener('click', () => {
            nav.classList.toggle('active');
            menuToggle.classList.toggle('open');
        });
    }
});
