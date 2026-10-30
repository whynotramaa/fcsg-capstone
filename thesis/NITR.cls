% ======================================================================
% STAND-IN for the college NITR.cls, which was not in the files we received.
% It implements the commands main.tex calls so the report builds today.
% When the department's NITR.cls arrives, overwrite this file with it and
% replace FrontPages.tex with the college version (keep our field values).
% ======================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{NITR}[2026/10/30 Stand-in NIT Rourkela report class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass{report}

\RequirePackage[a4paper,left=1.25in,right=1in,top=1in,bottom=1in,headheight=15pt]{geometry}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage{xcolor}

% ---- fields filled in FrontPages.tex
\newcommand{\@reporttitle}{Report Title}
\newcommand{\@doctype}{Project Report}
\newcommand{\@degree}{Bachelor of Technology}
\newcommand{\@dept}{Department}
\newcommand{\@authors}{Author}
\newcommand{\@supervisorone}{Supervisor}
\newcommand{\@supervisortwo}{Co-Supervisor}
\newcommand{\@submitdate}{\today}
\newcommand{\@dedicationtext}{}
\newcommand{\@acktext}{}
\newcommand{\@abstracttext}{}
\newcommand{\@keywords}{}
\newcommand{\reportTitle}[1]{\renewcommand{\@reporttitle}{#1}}
\newcommand{\docType}[1]{\renewcommand{\@doctype}{#1}}
\newcommand{\degree}[1]{\renewcommand{\@degree}{#1}}
\newcommand{\dept}[1]{\renewcommand{\@dept}{#1}}
\newcommand{\authors}[1]{\renewcommand{\@authors}{#1}}
\newcommand{\supervisorOne}[1]{\renewcommand{\@supervisorone}{#1}}
\newcommand{\supervisorTwo}[1]{\renewcommand{\@supervisortwo}{#1}}
\newcommand{\submitDate}[1]{\renewcommand{\@submitdate}{#1}}
\newcommand{\dedicationText}[1]{\renewcommand{\@dedicationtext}{#1}}
\newcommand{\ackText}[1]{\renewcommand{\@acktext}{#1}}
\newcommand{\abstractText}[1]{\renewcommand{\@abstracttext}{#1}}
\newcommand{\keywords}[1]{\renewcommand{\@keywords}{#1}}

\newcommand{\@institute}{National Institute of Technology Rourkela}
\newcommand{\@logo}{\includegraphics[width=1.25in]{NITlogo}}

% ---- front pages
\newcommand{\coverPage}{%
	\begin{titlepage}\centering\singlespacing
	\vspace*{0.4in}
	{\LARGE\bfseries \@reporttitle\par}
	\vspace{0.6in}
	{\large\@authors\par}
	\vspace{0.8in}
	\@logo\par
	\vspace{0.8in}
	{\large\@dept\par}
	{\large\bfseries \@institute\par}
	\vfill
	{\large\@submitdate\par}
	\end{titlepage}}

\newcommand{\titlePage}{%
	\begin{titlepage}\centering\singlespacing
	\vspace*{0.4in}
	{\LARGE\bfseries \@reporttitle\par}
	\vspace{0.4in}
	{\itshape \@doctype{} submitted in partial fulfillment\\
	of the requirements for the degree of\par}
	\vspace{0.15in}
	{\large\bfseries \@degree\par}
	\vspace{0.1in}
	{\itshape in\par}
	\vspace{0.1in}
	{\large\bfseries \@dept\par}
	\vspace{0.3in}
	{\itshape by\par}\vspace{0.1in}
	{\large\@authors\par}
	\vspace{0.3in}
	{\itshape under the supervision of\par}\vspace{0.1in}
	{\large\@supervisorone\par}
	\vfill
	\@logo\par\vspace{0.2in}
	{\bfseries \@dept\\ \@institute\par}
	\vspace{0.1in}{\@submitdate\par}
	\end{titlepage}
	\pagenumbering{roman}\setcounter{page}{1}}

\newcommand{\@frontchapter}[1]{%
	\chapter*{#1}\addcontentsline{toc}{chapter}{#1}\thispagestyle{plain}}

\newcommand{\@signline}[1]{%
	\begin{minipage}[t]{0.45\linewidth}\singlespacing
	\rule{0.9\linewidth}{0.4pt}\\ #1\end{minipage}}

\newcommand{\certificateOfExamination}{%
	\@frontchapter{Certificate of Examination}
	This is to certify that the work in the \@doctype{} entitled
	\emph{\@reporttitle} by \@authors{} has been examined and is accepted
	in partial fulfillment of the requirements for the degree of \@degree{}
	in \@dept.
	\vspace{1.2in}\par\noindent
	\@signline{Supervisor}\hfill\@signline{Examiner}
	\cleardoublepage}

\newcommand{\@certbody}{%
	This is to certify that the work presented in this \@doctype{} entitled
	\emph{\@reporttitle} by \@authors{} is a record of original research
	carried out under our supervision and guidance in partial fulfillment of
	the requirements for the degree of \@degree{} in \@dept. Neither this
	\@doctype{} nor any part of it has been submitted for any degree or
	diploma elsewhere.}

\newcommand{\certificateOneSupervisor}{%
	\@frontchapter{Supervisor's Certificate}
	\@certbody\vspace{1.2in}\par\noindent\hfill\@signline{\@supervisorone}
	\cleardoublepage}

\newcommand{\certificateTwoSupervisors}{%
	\@frontchapter{Supervisors' Certificate}
	\@certbody\vspace{1.2in}\par\noindent
	\@signline{\@supervisorone}\hfill\@signline{\@supervisortwo}
	\cleardoublepage}

\newcommand{\dedication}{%
	\thispagestyle{empty}\vspace*{2.5in}
	\begin{center}\itshape \@dedicationtext\end{center}
	\cleardoublepage}

\newcommand{\declaration}{%
	\@frontchapter{Declaration of Originality}
	We hereby declare that this \@doctype{} is our own work and that, to the
	best of our knowledge, it contains no material previously published or
	written by another person, nor material which has been accepted for the
	award of any other degree, except where due acknowledgement is made in
	the text. Results reported here are traceable to the project repository
	and its recorded experiment logs.
	\vspace{1.2in}\par\noindent\hfill\@signline{\@authors}
	\cleardoublepage}

\newcommand{\acknowledgment}{%
	\@frontchapter{Acknowledgement}
	\@acktext
	\cleardoublepage}

\renewcommand{\abstract}{%
	\@frontchapter{Abstract}
	\@abstracttext
	\ifx\@keywords\@empty\else\par\vspace{1em}\noindent\textbf{Keywords:} \@keywords\fi
	\cleardoublepage}

% Template commands not used by this report; defined so the original
% main.tex lines can be uncommented without errors.
\newcommand{\coverPageSynReg}{\coverPage}
\newcommand{\indexColumns}[1]{}
\newcommand{\indexPage}{}

\pagestyle{plain}
