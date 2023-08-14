% SPhdThesis v1.0
% By Saurabh Garg (saurabhgarg@mysoc.net)
% Version 1.0 released 11.06.2013

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bst/SPhdThesis_myversion}[2023/05/01 v1.0 SPhdThesis_myversion class]

\RequirePackage{xkeyval}

% \DeclareOption{copyright}{\@copyrighttrue}

% --------------------------------------------------------------------------
% Define option for formatting for [screen] and print.
% When formatting for screen colors are used for links, tables, and algorithms.
% When formatting for print black color is used for links, tables, and algorithms.
\define@choicekey*[Sg]{PhDThesis}{media}{screen,print}[screen]{\def \SgIntMedia{#1}}
\setkeys[Sg]{PhDThesis}{media}

% --------------------------------------------------------------------------
% Define option for formatting title page in lower or [upper] case.
\define@choicekey*[Sg]{PhDThesis}{titlecase}{upper,lower}[upper]{\def \SgIntTitleCase{#1}}
\setkeys[Sg]{PhDThesis}{titlecase}

% --------------------------------------------------------------------------
% Define line spacing. The valid values are [onehalf] and double.
% Note that before using \SgLineSpacing setspace package must be loaded.
\define@choicekey*[Sg]{PhDThesis}{linespacing}[\val\nr]{onehalf,double}[onehalf]{%
	\ifcase\nr\relax
		\def \SgIntLineSpacing{\onehalfspacing}
	\or
		\def \SgIntLineSpacing{\doublespacing}
	\fi
}
\setkeys[Sg]{PhDThesis}{linespacing}

% -----------------------------------------------------------------------------
% Define font size. The valid values are [11pt] and 12pt.
\define@choicekey*[Sg]{PhDThesis}{fontsize}{11pt,12pt}[11pt]{\def \SgIntFontSize{#1}}
\setkeys[Sg]{PhDThesis}{fontsize}

% -------------------------------------------------------------------------
% Define font family to be used. Default is cm.
\define@choicekey*[Sg]{PhDThesis}{font}{cm,times,helvet,palatino}[cm]{\def \SgIntFont{#1}}
\setkeys[Sg]{PhDThesis}{font}

% -------------------------------------------------------------------------
% Define open. The valid values are right and [any].
% Right makes chapters and entries in frontmatter begin  only on right hand pages
% or on the next page available. Any puts them on the next available page.
\define@choicekey*[Sg]{PhDThesis}{open}[\val\nr]{right,any}[any]{%
	\ifcase\nr\relax
		\def \SgIntOpen {openright}
	\or
		\def \SgIntOpen {openany}
	\fi
}
\setkeys[Sg]{PhDThesis}{open}

% -------------------------------------------------------------------------
% Define pageside. The valid values are oneside and [twoside].
% Specifies whether double or single sided output should be generated.
\define@choicekey*[Sg]{PhDThesis}{pageside}{oneside,twoside}[twoside]{\def \SgIntPageSide{#1}}
\setkeys[Sg]{PhDThesis}{pageside}


% Process all options defined above with default values.
\ProcessOptionsX[Sg]<PhDThesis>

% Finally load the report class.
\LoadClass[a4paper, \SgIntFontSize, \SgIntOpen, \SgIntPageSide]{report}




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F O N T S.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[T1]{fontenc}  % Use T1 encoded cm-super fonts.
\RequirePackage{microtype}    % Improve typesetting.
\RequirePackage{fix-cm}       % Support for arbitrary font size for cm.

% Specify the format for the section titles in toc.
\newcommand{\SgIntTocSectionFormat}{\bfseries}

% Specify the format for chapter name and number in chapter headings.
\newcommand{\SgIntChapNameFormat}{\fontsize{20}{50}\fontshape{sc}\selectfont}
\newcommand{\SgIntChapNumberFormat}{\fontsize{76}{80}\selectfont}

% Specify the format for chapter, section, and subsection titles.
\newcommand{\SgIntChapTitleFormat}{\LARGE\bfseries}
\newcommand{\SgIntSecTitleFormat}{\Large\bfseries}
\newcommand{\SgIntSubsecTitleFormat}{\large\bfseries}
\newcommand{\SgIntSubsubsecTitleFormat}{\large\bfseries}

% Specify the format for displaying chapter in fancy header.
\newcommand{\SgIntHeaderFormat}{\large\bfseries}
\newcommand{\SgIntPageNumFormat}{\bfseries}

% Set default font to sans-serif
\renewcommand{\familydefault}{\sfdefault}
% \fontfamily{Times New Roman}\selectfont

% Change captions fonts to PT Sans Narrow.
\newcommand{\ptsansnarrow}{\fontfamily{PTSansNarrow}\selectfont}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C O L O R S
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[usenames,dvipsnames,hyperref]{xcolor}
\definecolor{BlueGray}{rgb}{0.4, 0.6, 0.8}
\definecolor{Asparagus}{rgb}{0.53, 0.66, 0.42}
\definecolor{AuroMetalSaurus}{rgb}{0.43, 0.5, 0.5}
\definecolor{beaublue}{rgb}{0.74, 0.83, 0.9}
\definecolor{mygreen}{HTML}{E9ECE6}
\definecolor{webgreen}{rgb}{0,.5,0}
\definecolor{webblue}{rgb}{0,0,1}
\definecolor{Cadet}{rgb}{0.33, 0.41, 0.47}
\definecolor{Charcoal}{rgb}{0.21, 0.27, 0.31}
\definecolor{DarkElectricBlue}{rgb}{0.33, 0.41, 0.47}
\definecolor{Burgundy}{rgb}{0.5, 0.0, 0.13}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C A P T I O N S
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[labelfont=bf,font={small,color=Cadet}]{caption} %margin=10pt,


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  P A C K A G E S
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{graphicx, subfig}                     % Figures.
\RequirePackage{amsmath, amssymb, amsthm}             % Math symbols and fonts.
\RequirePackage{float}                                % Must be done before hyperref.
\RequirePackage{ifthen}                               % For comparison.
% \RequirePackage{etoolbox}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P A G E   L A Y O U T.
% Use geometry package to set up margins. \geometry{margin=2.5cm}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[letterpaper, left=2cm, right=2cm, top=2cm, bottom=2cm, includehead,\SgIntPageSide]{geometry}
% \RequirePackage[a4paper, left=1.25in, right=1in, top=1in, bottom=1in, includehead, \SgIntPageSide]{geometry}

\ifthenelse{\equal{\SgIntOpen}{openright}}
{
	% For adding extra blank page, if necessary, after chapter.
	\RequirePackage{emptypage}

	% This is used in frontmatter to add an empty page between ack, abstract, toc, lof and lot.
	\newcommand{\SgIntClearDoublePage}{\clearpage{\pagestyle{empty}\cleardoublepage}}
}
{ \newcommand{\SgIntClearDoublePage}{\clearpage} }

% For disabling paragraph indenting and using a blank line between paragraphs.
\RequirePackage{parskip}

% Set line spacing.
\RequirePackage{setspace}
\SgIntLineSpacing

% Fix footnote spacing
\setlength{\footnotesep}{0.5cm}   % Distance between two footnotes.
\setlength{\skip\footins}{0.5cm}  % Distance between last line of text and first footnote.

% By default Latex centers images vertically on a float page. Modify Latex internal variables so that figures are placed from top.
\makeatletter
    \setlength{\@fptop}{0.25cm}
    \setlength{\@fpsep}{1.00cm}
\makeatother

\setlength{\textfloatsep}{1.5cm} % Set the distance between a float and text.
\setlength{\floatsep}{1.0cm}     % Set the distance between two floats.

% Prevent latex from adding extra space between paragraphs so that the last line is at the bottom margin on each page.
\raggedbottom

% Define horizontal spacing between subfigures.
\newcommand{\SgIntHSpaceBetweenSubfloats}{\hspace{0.5cm}}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T A B L E
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{booktabs, colortbl} % Tables.
\RequirePackage{tabularx}           % Auto column sizing.

\renewcommand{\arraystretch}{1.5}     % Set space between rows in a table.
\renewcommand{\tabcolsep}{0.20cm}     % Set space between columns in a table.
\heavyrulewidth = 0.15em              % Set width of heavy rules.
\lightrulewidth = 0.07em              % Set width of light rules.
\abovetopsep    = 0.1cm               % Set separation between caption and top rule.
\aboverulesep   = 0.4ex               % Set separation to use above a rule.
\belowrulesep   = 0.4ex               % Set separation to use below a rule.

% Set color for table rules.
\ifthenelse{\equal{\SgIntMedia}{screen}}{\arrayrulecolor[rgb]{0.0, 0.6, 0.9}}{}
\ifthenelse{\equal{\SgIntMedia}{print}}{\arrayrulecolor[rgb]{0.0, 0.0, 0.0}}{}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A L G O R I T H M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[algo2e, algoruled, algosection, linesnumbered, noend]{algorithm2e}
% allowdisplaybreaks
\DontPrintSemicolon                      % Dont print semicolons at end of lines.
\algoheightrule       = \heavyrulewidth  % Set the width of the top and bottom rules.
\algotitleheightrule  = \lightrulewidth  % Set the width of the middle rule.
\SetAlgoInsideSkip{medskip}              % Set distance between middle rule and algorithm.
\interspacetitleruled = 0.2cm            % Set distance between caption and rules.
\setlength{\algomargin}{2.25em}          % Set the margin of the algorithm text.
\SetNlSkip{1.25em}                       % Set the spacing between line numbers and text.

\newenvironment{SgAlgorithm}[1][htbp]
{
	% \SetAlgorithmName{Method}{}{} % {name}{algotitle}{list of alg. title}
	\begin{algorithm2e}[#1]
	\SetAlgoLined
    \linespread{1.5}
    \selectfont
}
{\end{algorithm2e}}


%---------------------------------------------------------------
% S L E E K - T E M P L A T E
%---------------------------------------------------------------
\RequirePackage{bst/sleek-theorems}

%---------------------------------------------------------------
% T E X T  INSIDE  F R A M E
%---------------------------------------------------------------
\RequirePackage[framemethod=TikZ]{mdframed}

\newenvironment{mymdframed}[1]
{%
    \begin{mdframed}[
        frametitle=#1,
        nobreak=false,
        skipabove=\topskip,
        skipbelow=\topskip,
        innerleftmargin=0.5em,
        innerrightmargin=0.5em,
        innerbottommargin=0.5em,
        innertopmargin=0.5em,
        linewidth=0.25em,
        roundcorner=0.15em,
        linecolor=beaublue,
        frametitlebackgroundcolor=beaublue]
}
{\end{mdframed}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  B I B L I O G R A P H Y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Rename bibliography to references.
\renewcommand{\bibname}{References}

% Define a new command to include the bibliography file and
% set the formatting options.
\newcommand{\SgIncludeBibTex}[1]
{%
	\clearpage                                 % Fix the page number in TOC.
	\fancyhead{}							   % remove the heading chapter number
	\phantomsection                            % Fix the link in PDF.
	\addcontentsline{toc}{chapter}{References} % Add the bibliography to TOC.
	\bibliographystyle{abbrvnat}               % plainnat  abbrvnat unsrtnat
	\bibliography{#1}                          % Include the bibliography file.
}

\newcommand{\SgIncludeBibLatex}[1]
{%
	\clearpage                                 % Fix the page number in TOC.
	\fancyhead{}							   % remove the heading chapter number
	\phantomsection                            % Fix the link in PDF.
	\addcontentsline{toc}{chapter}{References} % Add the bibliography to TOC.
	\addbibresource{#1}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F A N C Y   H E A D E R
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{fancyhdr}

% Set page style to fancy.
\pagestyle{fancy}

% By default fancyhdr converts the chapter headings to uppercase,
% so restore the chapter casing.
\renewcommand{\chaptermark}[1]{\markboth{\chaptername\ \thechapter.\ #1}{}}

% Define header consisting of Chapter #. Chapter_Name on left side and
% page number on right side.
% \nouppercase is used on left header to force bibliography in lowercase.
\fancyhead{}
% \fancyhead[L]{\nouppercase{\SgIntHeaderFormat\leftmark}}
\fancyhead[R]{Chapter~\SgIntHeaderFormat\theHchapter}
\renewcommand{\headrulewidth}{0pt}
\setlength{\headheight}{14pt}

% Disable footer.
\fancyfoot{}
\fancyfoot[C]{\SgIntPageNumFormat\thepage}
\renewcommand{\footrulewidth}{0pt}

% Define header and footer for plain pages.
\fancypagestyle{plain}
{%
    % Disable header.
    \fancyhead{}
    \renewcommand{\headrulewidth}{0pt}

    % Footer contains the page number on right side.
    \fancyfoot{}
    \fancyfoot[C]{\SgIntPageNumFormat\thepage}
    \renewcommand{\footrulewidth}{0pt}
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T O C    L O F    L O T    L O A
% Change the appearance of toc, lof, lot, etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Using tocloft, the toc can be formatted easily.
\RequirePackage[titles, subfigure]{tocloft}

% Remove dots.
\renewcommand{\cftdotsep}{\cftnodots}

% Remove dots from list of algorithms.
% This is necessary because we use algorithm2e which mananges its own list of algorithms.
\makeatletter
	\renewcommand{\@dotsep}{5000}
\makeatother

% Format chapter entries differently in toc.
\renewcommand{\cftchapfont}{\SgIntTocSectionFormat}

% Fix the indentation of figure and table entries in the lof, lot, and loa.
\setlength{\cftfigindent}{0in}
\setlength{\cfttabindent}{0in}

\newcommand{\SgAddToc}{\tableofcontents\SgIntClearDoublePage}
\newcommand{\SgAddLof}{%
	\newpage
	\phantomsection % Requires hyperref; this is to fix the link.
	\addcontentsline{toc}{section}{\numberline{}\hspace{-.35in}{\SgIntTocSectionFormat{}List of Figures}}
	\listoffigures
	\SgIntClearDoublePage
}
\newcommand{\SgAddLot}{%
	\newpage
	\phantomsection % Requires hyperref; this is to fix the link.
	\addcontentsline{toc}{section}{\numberline{}\hspace{-.35in}{\SgIntTocSectionFormat{}List of Tables}}
	\listoftables
	\SgIntClearDoublePage
}
\newcommand{\SgAddLoa}{%
	\newpage
	\phantomsection % Requires hyperref; this is to fix the link.
	\addcontentsline{toc}{section}{\numberline{}\hspace{-.35in}{\SgIntTocSectionFormat{}List of Algorithms}}
	\listofalgorithmes % Note an extra e, it is required because we use algorithm2e.
	\SgIntClearDoublePage
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C H A P T E R   H E A D I N G
% Change the appearance of chapter headers and section titles.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change the fonts for chapter heading.
\RequirePackage[Lenny]{fncychap}
\ChNameVar{\SgIntChapNameFormat}
\ChNumVar{\SgIntChapNumberFormat}
\ChTitleVar{\SgIntChapTitleFormat}

% Set the format of the section and subsection titles.
\RequirePackage{sectsty}
\sectionfont{\SgIntSecTitleFormat}
\subsectionfont{\SgIntSubsecTitleFormat}
\subsubsectionfont{\SgIntSubsubsecTitleFormat}

% Adjust the spacing between titles and surrounding matter.
\RequirePackage{etoolbox}
\makeatletter
	% Reduce the space between chapter title and text.
	\patchcmd{\DOTI}{\vskip 40\p@}{\vskip 20\p@}{}{}
	\patchcmd{\DOTIS}{\vskip 40\p@}{\vskip 20\p@}{}{}% for unnumbered chapters

	% http://tex.stackexchange.com/questions/13357/fncychap-package-reduce-vertical-gap-space-between-header-and-chapter-heading
	% Reduce the spacing between top margin and chapter title.
	\renewcommand*{\@makechapterhead}[1]
	{%
		\vspace*{10\p@}
		{%
			\parindent \z@ \raggedright \normalfont
			\ifnum \c@secnumdepth >\m@ne
				\if@mainmatter % Fix for frontmatter, mainmatter, and backmatter 040920
					\DOCH
				\fi
			\fi
			\interlinepenalty\@M
			\if@mainmatter     % Fix for frontmatter, mainmatter, and backmatter 060424
				\DOTI{#1}%
			\else%
				\DOTIS{#1}%
			\fi
		}
	}

	% For the case \chapter*:
	\renewcommand*{\@makeschapterhead}[1]
	{%
		\vspace*{10\p@}%
		{%
			\parindent \z@ \raggedright \normalfont
			\interlinepenalty\@M \DOTIS{#1} \vskip 10\p@
		}
	}
\makeatother


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% H Y P E R E F
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Embedding URL's in document.
\RequirePackage[hyphens]{url}

% Enable hyperlinks only in the PDF.
\RequirePackage[bookmarks=true, pdfstartview=Fit, linktoc=page, pdfpagemode=UseNone]{hyperref}

% Define the colors of the hyperlink.
\ifthenelse{\equal{\SgIntMedia}{screen}}{%
	\hypersetup
	{%
		colorlinks = true,
		linkcolor  = webgreen,
		citecolor  = cyan,
		urlcolor   = webblue
	}
}{}

\ifthenelse{\equal{\SgIntMedia}{print}} { \hypersetup{colorlinks=false,pdfborder={0 0 0}} }{}

% Define the proerties describing PDF.
\newcommand{\SgIntSetupPdfProps}%
{\hypersetup
	{%
		pdfauthor   = \SgIntAuthor,
		pdftitle    = \SgIntTitle,
		pdfsubject  = \SgIntSubject,
		pdfkeywords = \SgIntKeywords
	}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C L E V E R E F
% Must come as late as possible, especially after hyperref.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[capitalize]{cleveref}

% Disable the automatic abbreviations of equations and figures.
\crefname{equation}{Equation}{Equations}
\crefname{figure}{Figure}{Figures}
\Crefname{equation}{Equation}{Equations}
\Crefname{figure}{Figure}{Figures}

% Change the way links are produced in PDF documents.
\crefformat{chapter}{#2Chapter~#1#3}
\crefformat{section}{#2Section~#1#3}
\crefformat{figure}{#2Figure~#1#3}
\crefformat{equation}{#2Equation~#1#3}
\crefformat{table}{#2Table~#1#3}
\Crefformat{chapter}{#2Chapter~#1#3}
\Crefformat{section}{#2Section~#1#3}
\Crefformat{figure}{#2Figure~#1#3}
\Crefformat{equation}{#2Equation~#1#3}
\Crefformat{table}{#2Table~#1#3}
\creflabelformat{equation}{#2#1#3}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% D O C U M E N T   P R O P E R T I E S
% DEFINITIONS NEEDED FOR PREFACE PAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newlength{\signwidth}
\setlength{\signwidth}{3.1in}
\newlength{\datewidth}
\setlength{\datewidth}{2.1in}

% Define empty variables for creating title and setting PDF properties.
\newcommand{\SgIntTitle}{}
\newcommand{\SgIntTitleNoBreaks}{}
\newcommand{\SgIntAuthor}{}
\newcommand{\SgIntAuthorDegrees}{}
\newcommand{\SgIntYear}{}
\newcommand{\SgIntSubject}{}
\newcommand{\SgIntKeywords}{}
\newcommand{\SgIntDegree}{}
\newcommand{\SgIntDepartment}{}
\newcommand{\SgIntUniversity}{}
\newcommand{\SgIntDeclarationDate}{}
\newcommand{\SgIntSupervisor}{}
\newcommand{\SgIntSuperTitle}{}
\newcommand{\SgIntSuperDept}{}
\newcommand{\SgIntSecondMember}{}
\newcommand{\SgIntThirdMember}{}
\newcommand{\SgIntFourthMember}{}


% Define commands for setting  the above variables.
\newcommand{\SgSetTitle}[1]{\renewcommand{\SgIntTitle}{#1}\SgIntSetupPdfProps}
\newcommand{\SgSetTitleNoBreaks}[1]{\renewcommand{\SgIntTitleNoBreaks}{#1}}
\newcommand{\SgSetAuthor}[1]{\renewcommand{\SgIntAuthor}{#1}\SgIntSetupPdfProps}
\newcommand{\SgSetAuthorDegrees}[1]{\renewcommand{\SgIntAuthorDegrees}{#1}}
\newcommand{\SgSetYear}[1]{\renewcommand{\SgIntYear}{#1}}
\newcommand{\SgSetSubject}[1]{\renewcommand{\SgIntSubject}{#1}\SgIntSetupPdfProps}
\newcommand{\SgSetKeywords}[1]{\renewcommand{\SgIntKeywords}{#1}\SgIntSetupPdfProps}
\newcommand{\SgSetDegree}[1]{\renewcommand{\SgIntDegree}{#1}}
\newcommand{\SgSetDepartment}[1]{\renewcommand{\SgIntDepartment}{#1}}
\newcommand{\SgSetUniversity}[1]{\renewcommand{\SgIntUniversity}{#1}}
\newcommand{\SgSetDeclarationDate}[1]{\renewcommand{\SgIntDeclarationDate}{#1}}
\newcommand{\SgSetSupervisor}[1]{\renewcommand{\SgIntSupervisor}{#1}}
\newcommand{\SgSetSuperTitle}[1]{\renewcommand{\SgIntSuperTitle}{#1}}
\newcommand{\SgSetSuperDept}[1]{\renewcommand{\SgIntSuperDept}{#1}}
\newcommand{\SgSetSecondMember}[1]{\renewcommand{\SgIntSecondMember}{#1}}
\newcommand{\SgSetThirdMember}[1]{\renewcommand{\SgIntThirdMember}{#1}}
\newcommand{\SgSetFourthMember}[1]{\renewcommand{\SgIntFourthMember}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T I T L E  P A G E
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\titlePageColor}[1]{{\color{BlueGray}\bfseries\expandafter{#1}}}

\ifthenelse{\equal{\SgIntTitleCase}{upper}}{ \newcommand{\SgIntMakeUpperCase}[1]{\MakeUppercase{#1}} }{ \newcommand{\SgIntMakeUpperCase}[1]{#1} }


% made based on ua-thesis.cls
\newcommand{\SgAddTitle}{%
	\thispagestyle{empty}
	\begin{center}
		\vspace*{0.5in}
		{\Large\uppercase\expandafter{\SgIntTitle}}
		\bigskip\smallskip\\
		\titlePageColor{By}
		\bigskip\smallskip\\
		{\large{\SgIntAuthor}}

		{\color{Asparagus}\parbox{2in}{\hrulefill}} \\
		{\color{AuroMetalSaurus}{\footnotesize Copyright~~\copyright~~{\SgIntAuthor}~~\SgIntYear}}

		\vspace{.4in}

		\titlePageColor{A Dissertation Submitted to the Faculty of the} \\
		\uppercase\expandafter{\SgIntDepartment}

		\vspace{.4in}

		\titlePageColor{In Partial Fulfillment of the Requirements} \\
		\titlePageColor{For the Degree of} \\
		DOCTOR OF PHILOSOPHY \\

		\vspace{.4in}

		\titlePageColor{In the Graduate College} \\
		\uppercase\expandafter{\SgIntUniversity}


		\vfill
		\SgIntDeclarationDate


	\end{center}
	\vspace*{0.3in}
	\SgIntClearDoublePage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPROVAL PAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\committeemember}[2]{%
	\if #1 ""
	\else
		\parbox[t]{\signwidth}{\centering\hrulefill\\{#1}{}{\\#2}}
		\hfill
		\parbox[t]{\datewidth}{\centering\hrulefill\\[-0.5ex] Date }
		\bigskip\bigskip\medskip%
	\fi
}

\newcommand{\SgAddApproval}{%
	\newpage
	\thispagestyle{plain}
	\begin{center}
		THE UNIVERSITY OF ARIZONA\\
		GRADUATE COLLEGE
	\end{center}
	\vspace{0.5cm}

	As members of the Dissertation Committee, we certify that we have read the dissertation prepared by {\SgIntAuthor}, titled \textit{\SgIntTitle}, and recommend that it be accepted as
	fulfilling the dissertation requirement for the Degree of Doctor of Philosophy.
	\vspace*{0.3in}

	\committeemember{\SgIntSupervisor}{}
	\committeemember{\SgIntSecondMember}{}
	\committeemember{\SgIntThirdMember}{}
	\committeemember{\SgIntFourthMember}{}


	\vfill
	\noindent
	Final approval and acceptance of this dissertation is contingent upon the candidate's submission of the final copies of the dissertation to the Graduate College.
	\bigskip\medskip\\
	I hereby certify that I have read this dissertation prepared under my direction and recommend that it be accepted as fulfilling the dissertation requirement.

	\committeemember{\SgIntSupervisor}{Dissertation Director}
	\newpage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STATEMENT BY AUTHOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\SgAddStatement}[1][nosignimage]{%
	\newpage
	\thispagestyle{plain}
	\begin{center}
		\vspace*{1.5cm}
		{ \Large \bfseries STATEMENT by AUTHOR }
	\end{center}

	\vspace{0.5cm}

	\begin{quote}
    This dissertation {\color{Cadet}\lowercase{\SgIntTitleNoBreaks}} prepared by {\color{Cadet}{\SgIntAuthor}} has been submitted in partial fulfillment of requirements for a doctoral degree at The University of Arizona
    and is deposited in the University Library to be made available to borrowers under the rules of the Library.

	Brief quotations from this dissertation are allowable without special permission, provided that accurate acknowledgment of the source is made. Requests for permission for extended quotation from or
    reproduction of this manuscript in whole or in part may be granted by
    % \if@copyright
	% 	the copyright holder.
    % \else
    %     the head of the major department or the Dean of the Graduate College when in his or her judgement the proposed use of the material is in the interests of scholarship. In all other instances,
    %     however, permission must be obtained from the author.
    % \fi
	\end{quote}

	% Checks if nosignimage. If so, it does nothing. Otherwise, it includes the signature image from the file specified by #1, shifted down by 0.5cm.
	\ifthenelse{\equal{#1}{nosignimage}}{}{\includegraphics[width=4cm, height=3cm]{#1}\vspace{-0.5cm}}
	SIGNED: {\color{Cadet}\underline{{\fontfamily{pzc}\selectfont\SgIntAuthor}}}

	\vspace*{0.5in}
	\newpage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ACKNOWLEDGMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\SgAddAcknowledgement}[1]{
    \begin{center}
    {\large\bfseries ACKNOWLEDGMENTS}
    \end{center}
    \bigskip
    \hspace*{\parindent}
    \input{#1}
    \newpage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEDICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{dedication}
    {\@dspacing\begin{center}}
    {\end{center}
    \newpage}

% \newcommand{\SgAddDeclaration}[1][nosignimage]{%
% 	\newpage
% 	\thispagestyle{plain}
% 	\begin{center}
% 	\vspace*{1.5cm}
% 	{ \Large \bfseries Declaration }
% 	\end{center}

% 	\vspace{0.5cm}

% 	\begin{quote}
% 	I hereby declare that this thesis is my original work and it has been written by me in its entirety. I have duly acknowledged all the sources of information which have been used in the thesis. \newline

% 	This thesis has also not been submitted for any degree in any university previously.
% 	\end{quote}

% 	\ifthenelse{\equal{#1}{nosignimage}}
% 	{ \vspace{2.0cm} }
% 	{ \vspace{0.5cm} }

% 	\begin{center}
% 		\ifthenelse{\equal{#1}{nosignimage}}
% 		{}{\includegraphics{#1}\\[-0.5cm]}
% 		\rule{5cm}{0.2mm} \\
% 		\SgIntAuthor \\
% 		\SgIntDeclarationDate
% 	\end{center}

% 	\SgIntClearDoublePage
% }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A B S T R A C T
% The abstract environment puts a large, bold, centered "Abstract" label at the top of the page. Abstract itself appears in a quote environment, i.e. tabbed in at both sides, and on its own page.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewenvironment{abstract}
{%
    \thispagestyle{plain}
	\begin{center}
		\vspace*{1.5cm}
		{\Large\bfseries Abstract}
	\end{center}
	\vspace{0.5cm}
	\begin{quote}
}
{%
    \end{quote}
	\SgIntClearDoublePage
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F R O N T M A T T E R
% The frontmatter environment set the page numbering to lowercase roman for
% ack, abstract, toc, lof, lot, loa, etc. It also resets page numbering for the
% remainder of thesis (arabic, starting at 1).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{frontmatter}
{%
	\setcounter{page}{1}
	\renewcommand{\thepage}{\roman{page}}
}
{%
	\clearpage
	\renewcommand{\thepage}{\arabic{page}}
	\setcounter{page}{1}
	\SgIntClearDoublePage
}

