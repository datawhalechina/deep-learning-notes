const configureMermaid = () => {
  if (typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    htmlLabels: false,
    fontFamily: "Arial, Microsoft YaHei, sans-serif",
    themeVariables: {
      fontSize: "18px",
      primaryColor: "#e9f2fc",
      primaryBorderColor: "#296bb7",
      primaryTextColor: "#333333",
      lineColor: "#0b0b0b",
    },
    themeCSS: `
      .node text,
      .nodeLabel,
      .label text {
        font-weight: 400;
        font-style: normal;
      }

      .node rect,
      .flowchart-link {
        stroke-width: 1px;
      }
    `,
  });

  if (typeof _quartoMermaid !== "undefined") {
    _quartoMermaid.resolveOptions = (svg) =>
      svg.closest(".cell")?.dataset ?? {};
  }
};

configureMermaid();
document.addEventListener("DOMContentLoaded", configureMermaid, { once: true });
