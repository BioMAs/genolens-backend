% Class for SINTEF letters with LaTeX.
%
% by Federico Zenith, federico.zenith@sintef.no.

\ProvidesClass{sciliciumletter}[2021/03/10 A class to write letters the SINTEF way]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{sciliciumdoc}}
\ProcessOptions\relax

\LoadClass{template/cls/sciliciumdoc}

% The next two lines come from the inerrant letter template of LaTeX.
\setlength\parskip{0.7em}
\setlength\parindent{0\p@}

% Set margins
\RequirePackage[top=4.75cm,        bottom=2.5cm,
                left=2.3cm,        right=2.3cm,
                headheight=4.25cm, headsep=5mm,
                footskip=2.5ex]{geometry}

% Setting up header and footer
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}% clear all header and footer fields
\lhead{\renewcommand{\baselinestretch}{0}%
       \raisebox{4.85mm}{\includegraphics[height=3.62cm]{\@mainlogo}}%
       }

% Header for first page
\fancypagestyle{firstpage}{
  \rhead{\begin{minipage}[b]{0.25\textwidth}%
    \@info%
    \end{minipage}%
  }
}

% Footer
\RequirePackage{lastpage}
\cfoot{%
  \sffamily%
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

% Recipient address and information colophon
\RequirePackage{setspace,tabularx}
\renewcommand{\@projectlabel}{%
  \EnglishFrench{Project / Reference}{Projet / Référence}}
\newcommand{\frontmatter}{%
  \thispagestyle{firstpage}%
  % Limit to 8cm so it fits in standard envelope window
  \begin{minipage}[t]{8cm}
    \vspace*{1ex}%
    \large%
    \onehalfspacing%
    \textbf{\@recipient}
  \end{minipage}

  \vspace{4ex}
  \begin{tabularx}{\textwidth}{@{}X@{}X@{}X@{}X@{}}%
    \@LabelText{\@yourreflabel} & \@LabelText{\@ourreflabel}
    & \@LabelText{\@projectlabel} & \@LabelText{\@datelabel}\\
    \@yourref & \@ourref & \@project & \@date
  \end{tabularx}%

}

% Closing
\newcommand*{\stopbreaks}{%
  \interlinepenalty\@M
  \def\par{\@@par\nobreak}%
  \let\\\@nobreakcr
  \let\vspace\@nobreakvspace}
\newcommand{\closing}[1]{\par\nobreak\vspace{\parskip}%
  \stopbreaks
  \noindent
  \parbox{\textwidth}{\raggedright
       \ignorespaces #1\\[6\medskipamount]%
       \ifx\@empty\fromsig
           \fromname
       \else \fromsig \fi\strut}%
   \par}
