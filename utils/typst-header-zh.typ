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

#let NotoSerif = (
    (name: "Libertinus Serif", covers: "latin-in-cjk"),
    "Noto Serif CJK SC",
)
#set text(
    font: NotoSerif,
    size: 12pt,
    fallback: false,
    top-edge: "bounds",
    bottom-edge: "bounds",
    lang: "en",
    region: "cn",
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
#show outline: set text(font: NotoSerif, fallback: false)
#let fit-image(it) = layout(size => {
    let image-width = measure(it).width
    let fitted = if image-width > size.width {
        scale(
            size.width / image-width * 100%,
            origin: top + left,
            reflow: true,
            it,
        )
    } else {
        it
    }
})
#show image: fit-image
#show figure: set align(center)
#show figure: set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)
#show raw: set text(font: ("JetBrains Mono", "Noto Serif CJK SC"), fallback: false)
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
