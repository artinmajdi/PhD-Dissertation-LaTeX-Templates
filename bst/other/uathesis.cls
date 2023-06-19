%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LATEX CLASS FILE FOR A UNIVERSITY OF ARIZONA THESIS
%
% Jeff Rodriguez
% rodriguez@ece.arizona.edu
% 20 Nov 95
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uathesis}[1995/11/20 University of Arizona Thesis Class]

\newif\if@PhD
\newif\if@copyright
\newif\if@draft
\DeclareOption{MS}{\@PhDfalse}
\DeclareOption{PhD}{\@PhDtrue}
\DeclareOption{copyright}{\@copyrighttrue}
\DeclareOption{draft}{\@drafttrue}
\DeclareOption{final}{\@draftfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
\PassOptionsToClass{12pt}{report}
\LoadClass{report}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MARGINS AND SPACING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\@sspacing}{\renewcommand{\baselinestretch}{1.0}\large\normalsize}
\newcommand{\@dspacing}{\renewcommand{\baselinestretch}{1.4}\large\normalsize}
\if@draft
    \newcommand{\@normalspacing}{\@sspacing}
\else
    \newcommand{\@normalspacing}{\@dspacing}
\fi
\@normalspacing

\setlength\topmargin		{    2pt}
\setlength\oddsidemargin	{ 41.8pt}
\setlength\evensidemargin	{    5pt}
\setlength\headheight		{   12pt}
\setlength\headsep		{ 24.5pt}
\setlength\topskip		{   12pt}
\setlength\textwidth		{420.8pt}
\setlength\textheight		{610.7pt}
\setlength\parskip		{    6pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PAGE NUMBERING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A macro that gives the date without the year.
\newcommand{\thedate}{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day}

\if@draft
    % Put page number at top right corner and add the date.
    \def\ps@topright{\let\@mkboth\@gobbletwo
        \def\@oddhead{\rm{\scriptsize\thedate}\hfil\thepage}\let\@evenhead
	\@oddhead\def\@evenfoot{}\def\@oddfoot{}}
    % Add the date.
    \def\ps@empty{\let\@mkboth\@gobbletwo
        \def\@oddhead{\rm{\scriptsize\thedate}\hfil}\let\@evenhead
	\@oddhead\def\@evenfoot{}\def\@oddfoot{}}
\else
    % Put page number at top right corner.
    \def\ps@topright{\let\@mkboth\@gobbletwo
        \def\@oddhead{\rm\hfil\thepage}\let\@evenhead\@oddhead
        \def\@evenfoot{}\def\@oddfoot{}}
\fi
\pagestyle{topright}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TABLE OF CONTENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use "TABLE OF CONTENTS" instead of "Contents" as the heading.
\renewcommand{\contentsname}{TABLE OF CONTENTS}

% Add dotted lines and remove \bfseries.
\renewcommand*{\l@chapter}[2]{%
    \ifnum \c@tocdepth >\m@ne
        \addpenalty{-\@highpenalty}%
	\if@draft
          \vskip 0.0em \@plus\p@
	\else
          \vskip 1.0em \@plus\p@
	\fi
        \setlength\@tempdima{1.5em}%
        \begingroup
            \parindent \z@ \rightskip \@pnumwidth
            \parfillskip -\@pnumwidth
            \leavevmode
            \advance\leftskip\@tempdima
            \hskip -\leftskip
            #1\nobreak
            \leaders\hbox{$\m@th \mkern \@dotsep mu.\mkern \@dotsep mu$}
            \hfil \nobreak\hbox to\@pnumwidth{\hss #2}\par
            \penalty\@highpenalty
        \endgroup
    \fi}

% Insert \@sspacing.
\def\@starttoc#1{\begingroup \makeatletter \@sspacing
    \@input{\jobname.#1}
    \if@filesw
	\expandafter\newwrite\csname tf@#1\endcsname
	\immediate\openout \csname tf@#1\endcsname \jobname.#1\relax
    \fi
    \global\@nobreakfalse \endgroup}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CAPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use \small, \@sspacing, and indent from both the left and right margins.
\newlength{\@capwidth}
\newlength{\@twoparindent}
\setlength{\@twoparindent}{\parindent}
\addtolength{\@twoparindent}{\parindent}
\long\def\@makecaption#1#2{%
  \@sspacing
  \vskip\abovecaptionskip
  \setlength{\@capwidth}{\hsize}
  \addtolength{\@capwidth}{-\@twoparindent}
  \sbox\@tempboxa{\small #1: #2}%
  \ifdim \wd\@tempboxa >\@capwidth
    \begin{center}{\parbox{\@capwidth}{\small #1: #2}}\end{center}
  \else
    \global \@minipagefalse
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip
  \@normalspacing}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FOOTNOTES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change to \@sspacing.
\long\def\@footnotetext#1{\insert\footins{%
    \@sspacing\reset@font\footnotesize
    \interlinepenalty\interfootnotelinepenalty
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth \@parboxrestore
    \protected@edef\@currentlabel{%
       \csname p@footnote\endcsname\@thefnmark
    }%
    \color@begingroup
      \@makefntext{%
        \rule\z@\footnotesep\ignorespaces#1\@finalstrut\strutbox}
    \color@endgroup}\@normalspacing}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LIST OF FIGURES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use "LIST OF FIGURES" instead of "List of Figures" as the heading.
\renewcommand{\listfigurename}{LIST OF FIGURES}

% Call \lofchapter* instead of \chapter*.
\renewcommand{\listoffigures}{%
    \if@twocolumn
        \@restonecoltrue\onecolumn
    \else
        \@restonecolfalse
    \fi
    \lofchapter*{\listfigurename
    \@mkboth{\uppercase{\listfigurename}}%
    {\uppercase{\listfigurename}}}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi}

% Add LOF to TOC.
% (Includes other below changes to \chapter.)
\newcommand{\lofchapter}{\if@openright\cleardoublepage\else\clearpage\fi
    \addcontentsline{toc}{chapter}{\listfigurename}
    \thispagestyle{topright}%
    \global\@topnum\z@
    \@afterindenttrue
    \secdef\@chapter\@schapter}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LIST OF TABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use "LIST OF TABLES" instead of "List of Tables" as the heading.
\renewcommand{\listtablename}{LIST OF TABLES}

% Call \lotchapter* instead of \chapter*.
\renewcommand{\listoftables}{%
    \if@twocolumn
        \@restonecoltrue\onecolumn
    \else
        \@restonecolfalse
    \fi
    \lotchapter*{\listtablename
    \@mkboth{\uppercase{\listtablename}}{\uppercase{\listtablename}}}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi}

% Add LOT to TOC.
% (Includes other below changes to \chapter.)
\newcommand{\lotchapter}{\if@openright\cleardoublepage\else\clearpage\fi
    \addcontentsline{toc}{chapter}{\listtablename}
    \thispagestyle{topright}%
    \global\@topnum\z@
    \@afterindenttrue
    \secdef\@chapter\@schapter}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECTIONING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Skip 30pt, rather than 40pt, after chapter headings.
\def\chapter@bottomskip{\vskip30pt}

% Change "Chapter" to "CHAPTER".
\renewcommand{\chaptername}{CHAPTER}

% Remove initial vspace, change \raggedright to \centering,
% change \huge & \Huge to \large, convert heading to uppercase, change vspace
% from 20pt to 7pt between "CHAPTER" and heading, and change final vspace to
% \chapter@bottomskip.
\def\@makechapterhead#1{%
    {\parindent \z@ \centering \reset@font
    \ifnum \c@secnumdepth >\m@ne
        \large\bfseries \@chapapp{} \thechapter
        \par\nobreak
        \vskip 7\p@
    \fi
    \interlinepenalty\@M
    \large \bfseries \uppercase\expandafter{#1}\par\nobreak
    \chapter@bottomskip}}

% Remove initial vspace, change \raggedright to \centering,
% change \Huge to \large, convert heading to uppercase, and change final vspace
% to \chapter@bottomskip.
\def\@makeschapterhead#1{%
    {\parindent \z@ \centering
    \reset@font
    \interlinepenalty\@M
    \large \bfseries \uppercase\expandafter{#1}\par\nobreak
    \chapter@bottomskip}}

% Use topright pagestyle instead of plain, and switch to \@afterindenttrue.
\renewcommand{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi
    \thispagestyle{topright}%
    \global\@topnum\z@
    \@afterindenttrue
    \secdef\@chapter\@schapter}

% Change \Large to \large.
\renewcommand{\section}{\@startsection {section}{1}{\z@}%
    {-3.5ex \@plus -1ex \@minus -.2ex}%
    {2.3ex \@plus.2ex}%
    {\reset@font\large\bfseries}}

% Change \large to \normalsize.
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\reset@font\normalsize\bfseries}}

% Switch to \afterindenttrue.
\def\@startsection#1#2#3#4#5#6{\if@noskipsec \leavevmode \fi
    \par \@tempskipa #4\relax
    \@afterindenttrue
    \ifdim \@tempskipa <\z@
        \@tempskipa -\@tempskipa \@afterindenttrue
    \fi
    \if@nobreak
	\everypar{}
    \else
        \addpenalty{\@secpenalty}\addvspace{\@tempskipa}
    \fi
    \@ifstar{\@ssect{#3}{#4}{#5}{#6}}%
    {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}}

% Convert chapter headings to uppercase.
\def\@chapter[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
        \refstepcounter{chapter}%
        \typeout{\@chapapp\space\thechapter.}%
        \addcontentsline{toc}{chapter}%
        {\protect\numberline{\thechapter}\uppercase\expandafter{#1}}%
    \else
        \addcontentsline{toc}{chapter}{#1}%
    \fi
    \chaptermark{\uppercase\expandafter{#1}}%
    \addtocontents{lof}{\protect\addvspace{10\p@}}%
    \addtocontents{lot}{\protect\addvspace{10\p@}}%
    \if@twocolumn
        \@topnewpage[\@makechapterhead{#2}]%
    \else
        \@makechapterhead{#2}%
        \@afterheading
    \fi}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPENDIX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change "Appendix" to "APPENDIX".
\renewcommand{\appendixname}{APPENDIX}

% Call \@appchapter instead of \@chapter.
% (Includes other above changes to \chapter.)
\renewcommand{\appendix}{\par
    \setcounter{chapter}{0}%
    \setcounter{section}{0}%
    \renewcommand{\@chapapp}{\appendixname}%
    \renewcommand{\thechapter}{\Alph{chapter}}
    \renewcommand{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi
    \thispagestyle{topright}%
    \global\@topnum\z@
    \@afterindenttrue
    \secdef\@appchapter\@schapter}}

% Change from "A" to "APPENDIX A" in the TOC.
% (Includes other above changes to \@chapter.)
\def\@appchapter[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
        \refstepcounter{chapter}%
        \typeout{\@chapapp\space\thechapter.}%
        \addcontentsline{toc}{chapter}%
        {\protect APPENDIX \space\thechapter:\space\uppercase\expandafter{#1}}%
    \else
        \addcontentsline{toc}{chapter}{#1}%
    \fi
    \chaptermark{\uppercase\expandafter{#1}}%
    \addtocontents{lof}{\protect\addvspace{10\p@}}%
    \addtocontents{lot}{\protect\addvspace{10\p@}}%
    \if@twocolumn
        \@topnewpage[\@makechapterhead{#2}]%
    \else
        \@makechapterhead{#2}%
        \@afterheading
    \fi}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REFERENCES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use "REFERENCES" instead of "Bibliography" as the heading.
\renewcommand{\bibname}{REFERENCES}

% Use \@sspacing
\renewenvironment{thebibliography}[1]
     {\@sspacing\chapter*{\bibname
        \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}}%
	\addcontentsline{toc}{chapter}{\bibname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist\@normalspacing}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFINITIONS NEEDED FOR PREFACE PAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\major#1{\gdef\@major{#1}}
\def\gradyear#1{\gdef\@gradyear{#1}}
\def\gradyearspaced#1{\gdef\@gradyearspaced{#1}}
\def\supervisor#1{\gdef\@supervisor{#1}}
\def\supertitle#1{\gdef\@supertitle{#1}}
\def\superdept#1{\gdef\@superdept{#1}}
\newlength{\signwidth}
\setlength{\signwidth}{3.1in}
\newlength{\datewidth}
\setlength{\datewidth}{2.1in}
\def\secondmember#1{\gdef\@secondmember{#1}}
\def\thirdmember#1{\gdef\@thirdmember{#1}}
\def\fourthmember#1{\gdef\@fourthmember{#1}}
\def\fifthmember#1{\gdef\@fifthmember{#1}}
\def\hrulefillthick{\leavevmode\leaders\hrule \@height 0.7pt \hfill\kern\z@}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TITLE PAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand{\maketitle}
    {\thispagestyle{empty}
    \vspace*{0.5in}	% shd be 1.0in, but that doesn't work for some reason
    \begin{center}
    \@dspacing
    {\large\bfseries\uppercase\expandafter{\@title}}
    \bigskip\smallskip\\
    \@sspacing
    {\bfseries by}
    \bigskip\smallskip\\
    {\large\bfseries{\@author}}
    \vfill
    \parbox{2.7in}{\hrulefillthick}
    \if@copyright
      \\
      {\footnotesize Copyright~~\copyright~~{\@author}~~\@gradyear}
    \fi
    \vfill
    A \if@PhD Dissertation \else Thesis \fi Submitted to the Faculty of the
    \bigskip\\
    DEPARTMENT OF {\uppercase\expandafter{\@major}}
    \bigskip\\
    In Partial Fulfillment of the Requirements\\
    For the Degree of
    \bigskip\\
    \if@PhD DOCTOR OF PHILOSOPHY \else MASTER OF SCIENCE \fi
    \bigskip\\
    In the Graduate College
    \bigskip\\
    THE UNIVERSITY OF ARIZONA
    \vfill
    {\@gradyearspaced}
    \end{center}
    \vspace*{0.3in}	% shd be 0.5in, but that doesn't work for some reason
    \newpage
    \@normalspacing
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPROVAL PAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\makeapproval}
    {\if@PhD
	\@sspacing
        \begin{center}
        THE UNIVERSITY OF ARIZONA\\
        GRADUATE COLLEGE
        \end{center}
        \bigskip
        As members of the Dissertation Committee, we certify that we have
	read the dissertation prepared by {\@author}, titled
	\textit{\@title}, and recommend that it be accepted as
	fulfilling the dissertation requirement for the Degree of
	Doctor of Philosophy.
	\bigskip\bigskip\medskip\\
	\parbox[t]{\signwidth}{\centering\hrulefillthick\\{\@supervisor}}
	\hfill
	\parbox[t]{\datewidth}{\centering\hrulefillthick\\Date}
        \if \@secondmember ""
	\else
	    \bigskip\bigskip\bigskip\medskip\\
	    \parbox[t]{\signwidth}{\centering\hrulefillthick\\{\@secondmember}}
	    \hfill
	    \parbox[t]{\datewidth}{\centering\hrulefillthick\\Date}
	    \bigskip\bigskip\bigskip\medskip\\
        \fi
        \if \@thirdmember ""
	\else
	    \parbox[t]{\signwidth}{\centering\hrulefillthick\\{\@thirdmember}}
	    \hfill
	    \parbox[t]{\datewidth}{\centering\hrulefillthick\\Date}
	    \bigskip\bigskip\bigskip\medskip\\
        \fi
        \if \@fourthmember ""
	\else
	    \parbox[t]{\signwidth}{\centering\hrulefillthick\\{\@fourthmember}}
	    \hfill
	    \parbox[t]{\datewidth}{\centering\hrulefillthick\\Date}
	    \bigskip\bigskip\bigskip\medskip\\
        \fi
        \if \@fifthmember ""
	\else
	    \parbox[t]{\signwidth}{\centering\hrulefillthick\\{\@fifthmember}}
	    \hfill
	    \parbox[t]{\datewidth}{\centering\hrulefillthick\\Date}
        \fi
	\bigskip
	\vfill
	\noindent
        Final approval and acceptance of this dissertation is contingent upon
        the candidate's submission of the final copies of the dissertation to
	the Graduate College.
        \bigskip\medskip\\
        I hereby certify that I have read this dissertation prepared under my
        direction and recommend that it be accepted as fulfilling the
        dissertation requirement.
	\bigskip\bigskip\medskip\\
        \parbox[t]{\signwidth}{
	    \centering\hrulefillthick\\{\@supervisor}\\Dissertation Director}
        \hfill
        \parbox[t]{\datewidth}{\centering\hrulefillthick\\Date}
        \newpage
	\@normalspacing
    \else
    \fi}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STATEMENT BY AUTHOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\makestatement}
    {\@sspacing
    \begin{center}
    {\large\bfseries STATEMENT BY AUTHOR}
    \end{center}
    \bigskip
    \hspace*{\parindent}
    This
    \if@PhD dissertation \else thesis \fi
    \textit{\@title}
    prepared by
    \@author
    has been submitted in partial fulfillment of
    requirements for a
    \if@PhD doctoral \else master's \fi
    degree at The University of Arizona
    and is deposited in the University Library to be made available to
    borrowers under the rules of the Library.
    \bigskip\\
    \hspace*{\parindent}
    Brief quotations from this
    \if@PhD dissertation \else thesis \fi
    are allowable without special
    permission, provided that accurate acknowledgment of the source is
    made. Requests for permission for extended quotation from or
    reproduction of this manuscript in whole or in part may be granted by
    \if@copyright
	the copyright holder.
	\bigskip
	\bigskip
    \else
        the head of the major department or the Dean of the Graduate
        College when in his or her judgement the proposed use of the
        material is in the interests of scholarship. In all other instances,
        however, permission must be obtained from the author.
    \fi
    \bigskip\smallskip\\
    \mbox{}\hfill SIGNED:\ \ \parbox[t]{\signwidth}{\hrulefillthick}
    \if@PhD
    \else
	\vfill
        \begin{center}
        {\large\bfseries APPROVAL BY THESIS DIRECTOR}
        \end{center}
        \bigskip
        \hspace*{\parindent}
        This thesis has been approved on the date shown below:
        \bigskip\bigskip\medskip\\
	\parbox[t]{\signwidth}{
	    \centering
	    \hrulefillthick\\
	    {\@supervisor}\\
	    {\@supertitle} of\\
	    {\@superdept}}
        \hfill
	\parbox[t]{\datewidth}{
	    \centering
	    \hrulefillthick\\
	    Date}
    \fi
    \vspace*{0.5in}
    \newpage
    \@normalspacing}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ACKNOWLEDGMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{acknowledgments}
    {\begin{center}
    {\large\bfseries ACKNOWLEDGMENTS}
    \end{center}
    \bigskip
    \hspace*{\parindent}}
    {\newpage}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEDICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{dedication}
    {\@dspacing\begin{center}}
    {\end{center}
    \@normalspacing
    \newpage}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ABSTRACT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewenvironment{abstract}
    {\chapter*{\abstractname}\addcontentsline{toc}{chapter}{ABSTRACT}}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UMI ABSTRACT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\makeumiabstract}
    {\if@PhD
	\thispagestyle{empty}
	\begin{center}
	{\large\bfseries{\@title}}
	\bigskip\bigskip\\
	{\@author}, Ph.D.\\
	The University of Arizona, {\@gradyear}
	\end{center}
	Director: {\@supervisor}
	\bigskip\bigskip\\
	\hspace*{\parindent}
	\input{abst}
	\setcounter{page}{0}
	\newpage
    \fi}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INCLUDE ONLY USER-SPECIFIED CHAPTERS AND RUN IN BATCH MODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\AtBeginDocument{%
    \typeout{}
    \typeout{File choices: \allthefiles}
    \typein[\typedfiles]{Enter files to be included (default = all files).}
    \if \typedfiles ""
        \renewcommand\typedfiles{\allthefiles}
        \typeout{Using all input files: \typedfiles.}
    \fi
    \includeonly{\typedfiles}
    \typeout{Running in batch mode.  See *.log file for error messages.}
    \batchmode}
