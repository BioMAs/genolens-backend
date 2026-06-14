% Class for Eurosafe reports with LaTeX.
%
% by Théophile Grébert, theophile@scilicium.com.
%
% Based on a class for SINTEF reports with LaTeX, by Federico Zenith, federico.zenith@sintef.no, found on OverLeaf

\ProvidesClass{eurosafereport}[2022/11/24 A class to prepare reports in Eurosafe style]

\newcommand*{\@summaryclassification}{\EnglishFrench{Restricted}{Restreint}}

\DeclareOption{unrestrictedsummary}{%%
  \renewcommand*{\@summaryclassification}{\EnglishFrench{Unrestricted}{Diffusion libre}}%
}
\DeclareOption{restrictedsummary}{%
  \renewcommand*{\@summaryclassification}{%
    \EnglishFrench{Restricted}{Restreint}
  }
}
\DeclareOption{confidentialsummary}{%
  \renewcommand*{\@summaryclassification}{%
    \EnglishFrench{Confidential}{Confidentiel}%
  }
}

% Should not be used in reports
\DeclareOption{internal}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{eurosafedoc}}
\ProcessOptions\relax


% Setting up alternating colors for table rows
\PassOptionsToPackage{table}{xcolor}


\LoadClass[report]{template/cls/eurosafedoc}

% Set margins
\RequirePackage[top=3cm,          bottom=3cm,
                left=2cm,         right=2cm,
                headheight=2.51cm, headsep=0.5cm,
                footskip=3ex]{geometry}

% Setting up link colors (in table of contents, and links in the document)
\RequirePackage{hyperref}

\hypersetup{
  pdftitle={\@pdftitle},
  pdfauthor={Eurosafe},
  colorlinks=true,
  linkcolor={eurosafedarkblue},
  filecolor={maroon},
  citecolor={blue},
  urlcolor={blue},
  pdfcreator={Eurosafe analysis pipeline}}

\newcommand{\@sponsorname}{}
\newcommand{\sponsorname}[1]{\renewcommand{\@sponsorname}{#1}}
\newcommand{\@sponsoraddress}{}
\newcommand{\sponsoraddress}[1]{\renewcommand{\@sponsoraddress}{#1}}

\newcommand{\@testfacilityname}{}
\newcommand{\testfacilityname}[1]{\renewcommand{\@testfacilityname}{#1}}
\newcommand{\@testfacilityaddress}{}
\newcommand{\testfacilityaddress}[1]{\renewcommand{\@testfacilityaddress}{#1}}

\newcommand{\@testsitename}{}
\newcommand{\testsitename}[1]{\renewcommand{\@testsitename}{#1}}
\newcommand{\@testsiteaddress}{}
\newcommand{\testsiteaddress}[1]{\renewcommand{\@testsiteaddress}{#1}}

\newcommand{\@sponsorlabel}{\EnglishFrench{Sponsor}{Sponsor}}
\newcommand{\@testfacilitylabel}{\EnglishFrench{Test Facility}{Installation d'Essai}}
\newcommand{\@testsitelabel}{\EnglishFrench{Test Site}{Site d'Essai}}


% Setting up headings/section colors
\RequirePackage{sectsty}% http://ctan.org/pkg/sectsty
\sectionfont{\color{eurosafedarkblue}}
\subsectionfont{\color{eurosafeblue}}
\subsubsectionfont{\color{eurosafedarkgray}}
% \sectionfont{\color{sciliciumsteelblue}}
% \subsectionfont{\color{sciliciumdarksteelblue}}
% \subsubsectionfont{\color{sciliciumdarkgrayblue}}
% \sectionfont{\color{sciliciumpoliceblue}}
% \subsectionfont{\color{sciliciumsky}}
% \subsubsectionfont{\color{sciliciumpearlviolet}}



% Setting up headers and footers
\RequirePackage{fancyhdr,lastpage}
\pagestyle{fancy}

% Standard header & footer
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}% clear all header and footer fields
\lhead{\renewcommand{\baselinestretch}{0}%
       \raisebox{0.2mm}{\includegraphics[height=2.34cm]{\@mainlogo}}%
       }
\cfoot{%
  \color{eurosafeblue}%
  \@separator\newline%
  \hspace*{\tabcolsep}%
  \begin{minipage}[c]{0.25\textwidth}
    \@LabelText \@projectlabel\newline
    \@project
  \end{minipage}%
  \begin{minipage}[c]{0.25\textwidth}
    \@LabelText \@reportnumberlabel\newline
    \@reportnumber
  \end{minipage}%
  \begin{minipage}[c]{0.25\textwidth}
    \@LabelText \@versionlabel\newline
    \@version
  \end{minipage}%
  \hfill
  \thepage\ \EnglishFrench{of}{sur} \PlainPageRef{LastPage}%
  \hspace*{\tabcolsep}\newline%
  \@separator
}

% plain style is conjured by first pages of new chapters
% empty definition makes it a clone of fancy
\fancypagestyle{plain}{}

\fancypagestyle{coverpage}{%
  \fancyhf{}% clear all header and footer fields
  \lhead{\renewcommand{\baselinestretch}{0}%
         \hspace*{2.3cm}\raisebox{4.85mm}%
         {\includegraphics[height=3.62cm]{\@mainlogo}}%
        }
}

\fancypagestyle{firstpage}{%
  \cfoot{%
    \color{eurosafeblue}
    \@separator\newline
    \begin{minipage}[c]{0.4\textwidth}
    ~\\\phantom{opera}
    \end{minipage}%
    \hfill
    \begin{minipage}[t]{0.4\textwidth}
      \flushright%
      \thepage\ \EnglishFrench{of}{sur} \PlainPageRef{LastPage}
    \end{minipage}%
    \hspace*{\tabcolsep}\newline%
    \@separator
  }
}

\fancypagestyle{backcover}{%
  \fancyhf{}% clear all header and footer fields
}

\newcommand{\@coverbackground}{}
\newcommand{\coverbackground}[1]{\renewcommand{\@coverbackground}{#1}}

\newcommand{\@clientlogo}{}
\newcommand{\clientlogo}[1]{\renewcommand{\@clientlogo}{#1}}




% Title page, first page, table of contents etc.
\RequirePackage{eso-pic,template/sty/eurosafehistory,template/sty/eurosafeattachments,tikz}
\newcommand{\frontmatter}{%


% -------- START FIRST PAGE --------
  \thispagestyle{firstpage}%
  \noindent%
  % Use 0.99\textheight as full \textheight can cause odd rounding errors.
  \begin{minipage}[t][0.99\textheight]{0.3\textwidth}
    \flushleft{\small\@info}
    \vspace{5ex}
    
    \vfill
  \end{minipage}%
  \begin{minipage}[t][0.99\textheight]{0.7\textwidth}
    \setlength{\parskip}{2ex}%
    \raggedright%
    \Huge\@reportlabel

    \vspace{2ex}
    \huge{\bfseries\@title}

    \normalsize \@subtitle

    \vspace{4ex}%
    \noindent%
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@versionlabel\\
      \@version
    \end{minipage}%
    \begin{minipage}[t]{0.47\textwidth}
      \@LabelText \@datelabel\\
      \@date
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.97\textwidth}
      \@LabelText \@Authorlabel\\
      \@author
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@clientlabel\\
      \@client
    \end{minipage}%
    \begin{minipage}[t]{0.47\textwidth}
      \@LabelText \@clientreflabel\\
      \@clientref
    \end{minipage}\\
    \@separator\\
    \ifcsempty{@sponsorname}{}{
        \hspace*{\tabcolsep}
        \begin{minipage}[t]{0.97\textwidth}
          \@LabelText \@sponsorlabel\\
          \@sponsorname\\
          \small\@sponsoraddress
        \end{minipage}\\
        \@separator\\
    }
    \ifcsempty{@testfacilityname}{}{
        \hspace*{\tabcolsep}
        \begin{minipage}[t]{0.97\textwidth}
          \@LabelText \@testfacilitylabel\\
          \@testfacilityname\\
          \small\@testfacilityaddress
        \end{minipage}\\
        \@separator\\
    }
    \ifcsempty{@testsitename}{}{
        \hspace*{\tabcolsep}
        \begin{minipage}[t]{0.97\textwidth}
          \@LabelText \@testsitelabel\\
          \@testsitename\\
          \small\@testsiteaddress
        \end{minipage}\\
        \@separator\\
    }
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@projectlabel\\
      \@project
    \end{minipage}%
    \begin{minipage}[t]{0.47\textwidth}
      \@LabelText \@pageslabel\\
      \PlainPageRef{LastPage}
      \@attachmentpages
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \vfill
	\@separator\\
	\hspace*{\tabcolsep}
	\begin{minipage}[t]{0.5\textwidth}
	  \@LabelText \@checkedlabel\\
	  \@checked
	\end{minipage}%
	\begin{minipage}[t]{0.47\textwidth}
      \@LabelText{\mdseries\@signaturelabel}\bigskip
	\end{minipage}\\
	\@separator\\
	\hspace*{\tabcolsep}
	\begin{minipage}[t]{0.5\textwidth}
	  \@LabelText \@approvedlabel\\
	  \@approved
	\end{minipage}%
	\begin{minipage}[t]{0.47\textwidth}
      \@LabelText{\mdseries\@signaturelabel}\bigskip
	\end{minipage}\\
	\@separator\\
	\hspace*{\tabcolsep}
    \begin{minipage}[t]{0.25\textwidth}
      \raggedright
      \@LabelText \@reportnumberlabel\\
      \@reportnumber
    \end{minipage}%
    \begin{minipage}[t]{0.25\textwidth}
      \ifbool{useISBN}{%
        \@LabelText{ISBN}\\
        \@ISBN%
      }{~}% Leave empty if no ISBN required
    \end{minipage}
  \end{minipage}


  \clearpage
  \null
  \clearpage
  \null
  \clearpage
  % \historypage

  \tableofcontents

  \attachmentsection
  
  \clearpage

  \noindent
}

\newcommand{\backmatter}{
  \clearpage%
  % Subtract one page so LastPage seems to refer to our intended last page.
  \addtocounter{page}{-1}%
  \thispagestyle{backcover}
  \pagecolor{eurosafeblue}%
  \vspace*{\fill}
  \centering
    \includegraphics[height=8cm,keepaspectratio]{%
         Eurosafe_logo_blanc_signature_path_\ifbool{print}{CMYK}{RGB}%
         }
  \vspace*{\fill}
}
