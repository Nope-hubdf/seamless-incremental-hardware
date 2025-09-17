import requests
from bs4 import BeautifulSoup
import csv
import os


# Synthesize ICML papers
def synthesize_icml(year, url, out_dir):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    papers = soup.find_all("div", class_="paper")
    results = []
    for paper in papers:
        title_tag = paper.find("p", class_="title")
        authors_tag = paper.find("span", class_="authors")
        if title_tag and authors_tag:
            title = title_tag.get_text(strip=True)
            authors = authors_tag.get_text(separator=",", strip=True).replace(
                "\xa0", " "
            )
            results.append([title, authors])
    out_path = os.path.join(out_dir, f"icml_{year}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["title", "authors"])
        writer.writerows(results)


# Synthesize NeurIPS papers
def synthesize_neurips(year, url, out_dir):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    papers = soup.find_all("li", class_="conference")
    out_path = os.path.join(out_dir, f"neurips_{year}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["title", "authors"])
        for paper in papers:
            a_tag = paper.find("a")
            i_tag = paper.find("i")
            if a_tag and i_tag:
                title = a_tag.get_text(strip=True)
                authors = i_tag.get_text(strip=True)
                writer.writerow([title, authors])


# Synthesize CVPR papers
def synthesize_cvpr(year, url, out_dir):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    titles = soup.find_all("dt", class_="ptitle")
    out_path = os.path.join(out_dir, f"cvpr_{year}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["title", "authors"])
        for title_tag in titles:
            a_tag = title_tag.find("a")
            if not a_tag:
                continue
            title = a_tag.get_text(strip=True)
            dd_tag = title_tag.find_next_sibling("dd")
            authors = []
            if dd_tag:
                author_forms = dd_tag.find_all("form", class_="authsearch")
                for form in author_forms:
                    author_a = form.find("a")
                    if author_a:
                        authors.append(author_a.get_text(strip=True))
            writer.writerow([title, ", ".join(authors)])


# Synthesize ACL/EMNLP papers (same logic)
def synthesize_acl_emnlp(conference, year, url, out_dir):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    papers = soup.find_all("span", class_="d-block")
    out_path = os.path.join(out_dir, f"{conference.lower()}_{year}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["title", "authors"])
        for paper in papers:
            strong_tag = paper.find("strong")
            title_tag = strong_tag.find("a") if strong_tag else None
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            author_tags = paper.find_all("a")[1:]
            authors = [a.get_text(strip=True) for a in author_tags]
            writer.writerow([title, ", ".join(authors)])


# Main synthesis dispatcher
def synthesize(conference, year):
    url = sources.get((conference, year))
    if not url:
        raise ValueError(f"No source URL for {conference} {year}")
    out_dir = os.path.join(os.path.dirname(__file__), "corpus")
    os.makedirs(out_dir, exist_ok=True)
    if conference == "ICML":
        synthesize_icml(year, url, out_dir)
    elif conference == "NeurIPS":
        synthesize_neurips(year, url, out_dir)
    elif conference == "CVPR":
        synthesize_cvpr(year, url, out_dir)
    elif conference in ["ACL", "EMNLP"]:
        synthesize_acl_emnlp(conference, year, url, out_dir)
    else:
        raise NotImplementedError(f"Synthesis for {conference} not implemented.")


sources = {
    ("ICML", 2024): "https://proceedings.mlr.press/v235/",
    ("NeurIPS", 2024): "https://papers.nips.cc/paper_files/paper/2024",
    ("CVPR", 2025): "https://openaccess.thecvf.com/CVPR2025?day=all",
    ("CVPR", 2024): "https://openaccess.thecvf.com/CVPR2024?day=all",
    ("ACL", 2025): "https://aclanthology.org/volumes/2025.acl-long/",
    ("ACL", 2024): "https://aclanthology.org/volumes/2024.acl-long/",
    ("EMNLP", 2024): "https://aclanthology.org/volumes/2024.emnlp-main/",
}

for conference, year in sources.keys():
    print(f"Synthesizing {conference} {year}...")
    synthesize(conference, year)
