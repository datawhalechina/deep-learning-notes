#import "@preview/showybox:2.0.4": showybox

#set document(
    title: "Deep Learning Notes",
    author: "jshn9515",
    description: "A collection of notes and code examples for deep learning concepts and techniques.",
    keywords: (
        "Deep Learning Tutorial",
        "PyTorch",
        "Quarto",
    ),
)
#set page(paper: "a4")

#set text(
    font: "Libertinus Serif",
    size: 12pt,
    fallback: false,
    top-edge: "bounds",
    bottom-edge: "bounds",
    lang: "en",
    region: "us",
)
#set par(
    first-line-indent: (amount: 2em, all: true),
    justify: true,
    spacing: 1em,
)
#set bibliography(style: "ieee")
#set heading(numbering: none)
#set figure(numbering: none)
#set table(align: center)
#set list(indent: 2em)
#set enum(indent: 2em)
#set math.equation(numbering: none)

#show heading: set par(first-line-indent: (amount: 0pt, all: false))
#show figure: set align(center)
#show figure: set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)
#show raw: set text(font: ("JetBrains Mono"), fallback: false)
#show raw.where(block: true): set text(size: 8pt)
#show table: it => align(center, it)
#show math.equation: set block(breakable: true)

#show quote.where(block: true): it => block(
  above: 1em,
  below: 1em,
)[
  #showybox(
    frame: (
      border-color: gray,
      thickness: 0.6pt,
      radius: 4pt,
    ),
    body-style: (
      fill: luma(248),
      inset: 1em,
    ),
  )[
    #it.body
  ]
]
