\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{SPhdThesis_myversion}[2023/05/01 v1.0 SPhdThesis_myversion class]

\RequirePackage{xkeyval}

% Define options for media, title case, line spacing, font size, font family, open and pageside.
\define@choicekey*[Sg]{PhDThesis}{media}{screen,print}[screen]{\def \SgIntMedia{#1}}
\define@choicekey*[Sg]{PhDThesis}{titlecase}{upper,lower}[upper]{\def \SgIntTitleCase{#1}}
\define@choicekey*[Sg]{PhDThesis}{linespacing}[\val\nr]{onehalf,double}[onehalf]{%
	\ifcase\nr\relax
		\def \SgIntLineSpacing{\onehalfspacing}
	\or
		\def \SgIntLineSpacing{\doublespacing}
	\fi
}
\define@choicekey*[Sg]{PhDThesis}{fontsize}{11pt,12pt}[11pt]{\def \SgIntFontSize{#1}}
\define@choicekey*[Sg]{PhDThesis}{font}{cm,times,helvet,palatino}[cm]{\def \SgIntFont{#1}}
\define@choicekey*[Sg]{PhDThesis}{open}[\val\nr]{right,any}[any]{%
	\ifcase\nr\relax
		\def \SgIntOpen {openright}
	\or
		\def \SgIntOpen {openany}
	\fi
}
\define@choicekey*[Sg]{PhDThesis}{pageside}{oneside,twoside}[twoside]{\def \SgIntPageSide{#1}}

\ProcessOptionsX[Sg]<PhDThesis>
\LoadClass[a4paper, \SgIntFontSize, \SgIntOpen, \SgIntPageSide]{report}

% Fonts and Formatting
\RequirePackage[T1]{fontenc}
\RequirePackage{microtype}
\RequirePackage{fix-cm}

% Colors
\RequirePackage[usenames,dvipsnames,hyperref]{xcolor}

% Captions
\RequirePackage{caption}

% Packages
\RequirePackage{graphicx, subfig, amsmath, amssymb, amsthm, float, ifthen}

% Page Layout
\RequirePackage[letterpaper, left=2cm, right=2cm, top=2cm, bottom=2cm, includehead,\SgIntPageSide]{geometry}
\ifthenelse{\equal{\SgIntOpen}{openright}}{\RequirePackage{emptypage}}{}
\RequirePackage{parskip}
\RequirePackage{setspace}
\SgIntLineSpacing

% Tables and Algorithms
\RequirePackage{booktabs, colortbl, tabularx, algorithm2e}

% Sleek Template
\RequirePackage{bst/sleek-theorems}

% Text Inside Frame
\RequirePackage[framemethod=TikZ]{mdframed}

% Bibliography
\renewcommand{\bibname}{References}

% Fancy Header
\RequirePackage{fancyhdr}

% TOC, LOF, LOT, LOA
\RequirePackage[titles, subfigure]{tocloft}

% Chapter Heading
\RequirePackage[Lenny]{fncychap}
\RequirePackage{sectsty}
\RequirePackage{etoolbox}

% Hyperref
\RequirePackage[hyphens]{url}
\RequirePackage[bookmarks=true, pdfstartview=Fit, linktoc=page, pdfpagemode=UseNone]{hyperref}

% Cleveref
\RequirePackage[capitalize]{cleveref}

% Document Properties
\definecolor{BlueGray}{rgb}{0.4, 0.6, 0.8}
\definecolor{Asparagus}{rgb}{0.53, 0.66, 0.42}
\definecolor{AuroMetalSaurus}{rgb}{0.43, 0.5, 0.5}
