% Class for Scilicium reports with LaTeX.
%
% by Théophile Grébert, theophile@scilicium.com
%
% Based on class for SINTEF status reports with LaTeX,
% by Federico Zenith, federico.zenith@sintef.no.

\ProvidesClass{sciliciumstatus}[2021/11/01 A class to prepare status reports in 
the Scilicium style]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{sciliciumdoc}}
\ProcessOptions\relax

\LoadClass{template/cls/sciliciumdoc}

% Set margins
\RequirePackage[top=4.75cm,        bottom=2.5cm,
                left=2.3cm,        right=2.3cm,
                headheight=4.25cm, headsep=5mm,
                footskip=1ex]{geometry}

% Setting up header and footer
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}% clear all header and footer fields
\lhead{\renewcommand{\baselinestretch}{0}%
       \raisebox{4.85mm}{\includegraphics[height=3.62cm]{\@mainlogo}}%
       }
% Footer
\RequirePackage{lastpage}
\cfoot{%
  \color{sciliciumgray}%
  \@separator\newline
  \hspace*{\tabcolsep}%
  \begin{minipage}[c]{0.5\textwidth}
    \@LabelText\@projectlabel\newline
    \@project
  \end{minipage}
  \hfill
  \thepage\ \EnglishFrench{of}{sur} \PlainPageRef{LastPage}
  \hspace*{\tabcolsep}\newline
  \@separator
}
% Header for first page
\fancypagestyle{firstpage}{
  \rhead{\begin{minipage}[b]{0.25\textwidth}%
    \@info%
    \end{minipage}%
  }
}

% Definition of variables
\heading{\EnglishFrench{Status Report}{Rapport de statut}}

\newcommand*{\@plannedexpenditures}{}
\newcommand*{\plannedexpenditures}[1]{%
  \renewcommand{\@plannedexpenditures}{%
    \addplot[cyan, very thick, mark=diamond*] coordinates { #1 };%
  }%
}
\newcommand {\@accumulatedexpenditures}{}
\newcommand*{\accumulatedexpenditures}[1]{%
  \renewcommand{\@accumulatedexpenditures}{%
    \addplot[sciliciumblue, very thick, mark=square*] coordinates { #1 };%
  }%
}

\RequirePackage{tabularx,pgfplots}
\pgfplotsset{compat=1.3}

\RequirePackage{datetime2-calc}
% Define commands for short year number and short month name
\newcommand{\ignoretwo}[2]{}
\newcommand{\shortyear}{\expandafter\ignoretwo\year}
\providecommand*{\DTMbritishshortMonthname}[1][\month]{%
\ifcase#1\relax
\or Jan \or Feb \or Mar \or Apr \or May \or Jun%
\or Jul \or Aug \or Sep \or Oct \or Nov \or Dec%
\fi}
\providecommand*{\DTMnorskshortMonthname}[1][\month]{%
\ifcase#1\relax
\or Jan \or Fev \or Mar \or Avr \or Mai \or Jun%
\or Jul \or Aou \or Sep \or Oct \or Nov \or Dec%
\fi}

\pgfkeys{
  % small skip for thousands separator, no comma
  /pgf/number format/set thousands separator={\,},
  % group-four-digits=false
  /pgf/number format/min exponent for 1000 sep=4
}
\usepgfplotslibrary{dateplot}

% Recipient address and information colophon
\newcommand{\frontmatter}{%
  \thispagestyle{firstpage}%
  \noindent%
  \begin{minipage}[b]{0.7\textwidth}
    \setlength{\parskip}{2ex}%
    \Huge \@heading

    \huge\textbf \@title
  \end{minipage}

  \vspace{4ex}%
  \noindent%
  \@separator\\
  \hspace*{\tabcolsep}
  \begin{minipage}[t]{0.48\textwidth}
    \@LabelText \@clientlabel\\
    \@client
  \end{minipage}\hfill
  \begin{minipage}[t]{0.5\textwidth}
    \@LabelText \@clientreflabel\\
    \@clientref
  \end{minipage}\\
  \@separator\\
  \hspace*{\tabcolsep}
  \begin{minipage}[t]{0.48\textwidth}
    \@LabelText \@unitlabel\\
    \@institute
  \end{minipage}\hfill
  \begin{minipage}[t]{0.5\textwidth}
    \@LabelText \@datelabel\\
    \@date
  \end{minipage}\\
  \@separator\\
  \hspace*{\tabcolsep}
  \begin{minipage}[t]{0.48\textwidth}
    \@LabelText \@projectlabel\\
    \@project
  \end{minipage}\hfill
  \begin{minipage}[t]{0.5\textwidth}
    \@LabelText \@statusdatelabel\\
    \@statusdate
  \end{minipage}\\
  \@separator\\
  \hspace*{\tabcolsep}
  \begin{minipage}[t]{0.48\textwidth}
  \@LabelText \@classificationlabel\\
  \@classification
  \end{minipage}\\
  \@separator\\
  \begin{footnotesize}%
  \begin{tabular*}{\textwidth}{p{0.29\textwidth}p{0.27\textwidth}
                               p{0.2\textwidth}>{\centering}p{0.135\textwidth}}
    \@LabelText\@expenditurelabel & \@LabelText\@elapsedlabel & 
    \@LabelText\@statuslabel & \@LabelText\@asplannedlabel\tabularnewline
    \multicolumn{4}{@{}c@{}}{\@separator}\tabularnewline
    \@lastperiodlabel \hfill \@lastexp  & \@lastperiodlabel \hfill \@lasthrs  & 1. \@academiclabel & \@academic   \tabularnewline
    \@totallabel      \hfill \@totalexp & \@totallabel      \hfill \@totalhrs & 2. \@schedulelabel & \@onschedule \tabularnewline
    \@wageslabel      \hfill \@wages    &                                   & 3. \@expshortlabel & \@onbudget   \tabularnewline
    \\
    \@planexplabel    \hfill \@planexp  & & \multicolumn{2}{l}{\@attachmentrequest}
  \end{tabular*}
  \@separator
  \begin{center}
    \begin{tikzpicture}
      \begin{axis}[height=7cm,
                  width=0.75\textwidth,
                  title =\textbf{\@expshortlabel},
                  xlabel near ticks,
                  xlabel=\textbf{\@timelabel},
                  ylabel=\textbf{\@currency},
                  axis x line*=bottom,
                  axis y line*=left,
                  scaled ticks=false,
                  date coordinates in=x,
                  x tick label style={rotate=90},
                  xticklabel=\DTMshortMonthname\ \shortyear,
                  yticklabel={$\mathsf{\pgfmathprintnumber{\tick}}$},
                  xtick=data,
                  xtick align=outside,
                  ytick align=outside,
                  ymajorgrids,
                  legend style={at={(1.02, 0.5)}, anchor=west, draw=none}]
        \@plannedexpenditures
        \@accumulatedexpenditures
        \legend{\@planlabel, \@accumulatedlabel}
      \end{axis}
    \end{tikzpicture}
  \end{center}
  \@separator\\
  \hspace*{\tabcolsep}
  \begin{minipage}[t]{0.48\textwidth}
    \@LabelText \@managerlabel\\
    \@manager
  \end{minipage}\hfill
  \begin{minipage}[t]{0.5\textwidth}
    \MakeUppercase{\@signaturelabel}\bigskip
  \end{minipage}\\
  \@separator
  \end{footnotesize}
  \clearpage
}
