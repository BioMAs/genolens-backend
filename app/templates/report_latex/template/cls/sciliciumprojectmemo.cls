% Class for SINTEF memos with LaTeX.
%
% by Federico Zenith, federico.zenith@sintef.no.

\ProvidesClass{sciliciumprojectmemo}[2021/11/22 A class to prepare project memos 
in scilicium]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{sciliciumdoc}}
\ProcessOptions\relax
\LoadClass{template/cls/sciliciumdoc}

% Set margins
\RequirePackage[top=4.75cm,        bottom=3cm,
                left=1.8cm,        right=2.3cm,
                headheight=4.25cm, headsep=5mm,
                footskip=2.5ex]{geometry}

\heading{\EnglishFrench{Project Memo}{Mémo projet}}

% Setting up header and footer
\RequirePackage{fancyhdr,lastpage}
\pagestyle{fancy}


% Header
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}% clear all header and footer fields
\lhead{\renewcommand{\baselinestretch}{0}%
       \raisebox{4.85mm}{\includegraphics[height=3.62cm]{\@mainlogo}}
       }

% Footer
\cfoot{%
  \color{sciliciumgray}%
  \@separator\newline%
  \hspace*{\tabcolsep}
  \begin{minipage}[c]{0.25\textwidth}
    \@LabelText \@projectlabel\newline
    \@project
  \end{minipage}%
  \begin{minipage}[c]{0.25\textwidth}
    \@LabelText \@projectmemolabel\newline
    \@projectmemo
  \end{minipage}%
  \begin{minipage}[c]{0.25\textwidth}
    \@LabelText \@versionlabel\newline
    \@version
  \end{minipage}\hfill
  \thepage\ \EnglishFrench{of}{sur} \PlainPageRef{LastPage}
  \hspace*{\tabcolsep}\newline
  \@separator
}

\fancypagestyle{firstpage}{%
  \chead{%
    \raggedright\hspace*{.25\textwidth}\raisebox{1pt}{\Huge \@heading}%
  }
  \cfoot{%
    \color{sciliciumgray}%
    \@separator\newline
    \hspace*{\tabcolsep}
    \begin{minipage}[c]{0.25\textwidth}
      \phantom{\@LabelText \@projectlabel\newline}
      \phantom{\@project}
    \end{minipage}\hfill
    \thepage\ \EnglishFrench{of}{sur} \PlainPageRef{LastPage}%
    \hspace*{\tabcolsep}\newline
    \@separator
  }
}



% Recipient address and information colophon
\RequirePackage{setspace,template/sty/sciliciumhistory,template/sty/sciliciumattachments}
\newcommand{\frontmatter}{%
  \thispagestyle{firstpage}%
  \noindent%
  % Use 0.99\textheight as full \textheight can cause odd rounding errors.
  \begin{minipage}[t][0.99\textheight]{0.25\textwidth}
    \@info
    \vfill
    \includegraphics[width=\textwidth]{isoinscription}
  \end{minipage}%
  \begin{minipage}[t][0.99\textheight]{0.75\textwidth}
    \setlength{\parskip}{2ex}%
    \vspace{2ex}
    \huge\textbf \@title\\
    \normalsize \@subtitle

    \vspace{4ex}%
    \noindent%
    \raggedright
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.48\textwidth}
      \@LabelText \@versionlabel\\
      \@version
    \end{minipage}%
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@datelabel\\
      \@date
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.98\textwidth}
      \@LabelText \@authorlabel\\
      \@author
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.48\textwidth}
      \@LabelText \@clientlabel\\
      \@client
    \end{minipage}%
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@clientreflabel\\
      \@clientref
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.48\textwidth}
      \@LabelText \@projectlabel\\
      \@project
    \end{minipage}%
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@pageslabel\\
      \PlainPageRef{LastPage} \@attachmentpages
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.98\textwidth}
      {\large\bfseries\@abstractlabel}\\
      \@abstract
    \end{minipage}\\
    \vfill
    \@separator\\
    \hspace*{\tabcolsep}
	\begin{minipage}[t]{0.48\textwidth}
	  \@LabelText \@preparedlabel\\
	  \@prepared
	\end{minipage}%
	\begin{minipage}[t]{0.5\textwidth}
	  \@LabelText\textmd\@signaturelabel\bigskip
	\end{minipage}\\
	\@separator\\
	\hspace*{\tabcolsep}
	\begin{minipage}[t]{0.48\textwidth}
	  \@LabelText \@approvedlabel\\
	  \@approved
	\end{minipage}%
	\begin{minipage}[t]{0.5\textwidth}
	  \@LabelText\textmd\@signaturelabel\bigskip
	\end{minipage}\\
	\@separator\\
	\hspace*{\tabcolsep}
    \begin{minipage}[t]{0.48\textwidth}
      \@LabelText \@projectmemolabel\\
      \@projectmemo
    \end{minipage}%
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@classificationlabel\\
      \@classification
    \end{minipage}
  \end{minipage}

  \clearpage

  \historypage

  \tableofcontents

  \attachmentsection

  \clearpage

  \noindent
}
