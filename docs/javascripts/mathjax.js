window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};

function typesetMath() {
  if (!window.MathJax || !window.MathJax.typesetPromise) {
    return;
  }

  window.MathJax.typesetPromise().catch((error) => {
    console.error("MathJax typeset failed:", error);
  });
}

function scheduleTypeset(attempt = 0) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    typesetMath();
    return;
  }

  if (attempt < 15) {
    setTimeout(() => {
      scheduleTypeset(attempt + 1);
    }, 200);
  }
}

if (!window.MathJax.Hub) {
  window.MathJax.Hub = {
    Config: () => { },
    Queue: () => { }
  };
}

document.addEventListener("DOMContentLoaded", () => {
  scheduleTypeset();
});

window.addEventListener("load", () => {
  scheduleTypeset();
});

if (typeof window.document$ !== "undefined") {
  window.document$.subscribe(() => {
    scheduleTypeset();
  });
}
