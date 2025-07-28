import asyncio
import os
import json
from urllib.parse import urlparse

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
import requests
import fitz  # PyMuPDF
import re
from collections import defaultdict
import numpy as np

# ---------------------- Step 1: Save URL as PDF ----------------------

async def save_url_as_pdf(url, output_path="output.pdf"):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=120000, wait_until="networkidle")
            await page.pdf(path=output_path, format="A4", print_background=True)
            await browser.close()
            print(f"✅ Saved {url} as {output_path}")
            return True
    except PlaywrightTimeoutError:
        print(f"❌ Timeout while saving {url} as PDF.")
    except Exception as e:
        print(f"❌ Error saving {url} as PDF: {e}")
    return False

# ---------------------- Step 2: Extract HTML Outline ----------------------

def extract_outline_from_html(url, start_page=0):
    try:
        response = requests.get(url, timeout=20)
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"❌ Failed to fetch HTML outline from {url}: {e}")
        return {
            "title": "Untitled Webpage",
            "outline": []
        }

    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Webpage"
    outline = []
    for level in range(1, 7):  # h1 to h6
        for tag in soup.find_all(f'h{level}'):
            text = tag.get_text(strip=True)
            if not text:
                continue
            outline.append({
                "level": f"H{level}",
                "text": text,
                "page": start_page
            })

    return {
        "title": title,
        "outline": outline
    }

# ---------------------- Step 3: Extract Features from PDF ----------------------

from collections import defaultdict
import numpy as np

def extract_pdf_features(pdf_path):
    result = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Error opening PDF {pdf_path}: {e}")
        return result

    all_font_sizes = []

    # First pass: collect all font sizes
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    all_font_sizes.append(round(span["size"], 2))

    if not all_font_sizes:
        print(f"❌ No font sizes found in {pdf_path}")
        return result

    # Z-score normalization
    sizes_np = np.array(all_font_sizes)
    mean = np.mean(sizes_np)
    std = np.std(sizes_np) if np.std(sizes_np) != 0 else 1
    print(f"📏 Z-score normalization: mean={mean:.2f}, std={std:.2f}")

    # Second pass: extract + normalize
    for page_num, page in enumerate(doc):
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    normalized_size = round((span["size"] - mean) / std, 3)
                    spans.append({
                        "text": span["text"],
                        "size": normalized_size,  # replace original size
                        "flags": span["flags"],
                        "font": span["font"],
                        "bbox": span["bbox"],
                        "origin": (span["bbox"][0], span["bbox"][1])
                    })

        spans.sort(key=lambda s: (round(s["origin"][1]), s["origin"][0]))

        # Merge spans into lines
        lines, current_line, current_y = [], [], None
        y_threshold = 2
        for span in spans:
            y = round(span["origin"][1])
            if current_y is None or abs(y - current_y) <= y_threshold:
                current_line.append(span)
                current_y = y
            else:
                lines.append(merge_line(current_line))
                current_line = [span]
                current_y = y
        if current_line:
            lines.append(merge_line(current_line))

        # Merge lines into paragraphs
        paragraphs = []
        if lines:
            current_para = lines[0].copy()
            for i in range(1, len(lines)):
                same_font = lines[i]["font"] == lines[i - 1]["font"]
                same_size = lines[i]["size"] == lines[i - 1]["size"]
                if same_font and same_size:
                    current_para["text"] += " " + lines[i]["text"]
                    current_para["bbox"][3] = max(current_para["bbox"][3], lines[i]["bbox"][3])
                else:
                    paragraphs.append(current_para)
                    current_para = lines[i].copy()
            paragraphs.append(current_para)

        result.append({
            "page_number": page_num,
            "width": page.rect.width,
            "height": page.rect.height,
            "text_blocks": paragraphs
        })

    return result


def merge_line(spans):
    if not spans:
        return {}
    spans.sort(key=lambda s: s["origin"][0])
    full_text = "".join(span["text"] for span in spans).strip()

    # Clean trailing digits and excessive dots
    full_text = re.sub(r'\d+$', '', full_text).rstrip()
    full_text = re.sub(r'[.-]{4,}$', '', full_text).rstrip()

    return {
        "text": full_text,
        "size": round(max(s["size"] for s in spans), 3),  # this is now normalized
        "flags": spans[0]["flags"],
        "font": spans[0]["font"],
        "bbox": [
            spans[0]["bbox"][0],
            spans[0]["bbox"][1],
            spans[-1]["bbox"][2],
            max(s["bbox"][3] for s in spans)
        ]
    }

# ---------------------- Step 4: Run Entire Pipeline ----------------------

async def run_pipeline(urls):
    os.makedirs("pdfs", exist_ok=True)
    os.makedirs("outlines", exist_ok=True)
    os.makedirs("jsondata", exist_ok=True)

    total_data = []
    json_path = f"jsondata/bestdata2.json"
    try:
        with open(json_path, "r", encoding="utf8") as f:
            prevdata = json.load(f)
    except:
        prevdata = []

    for i, url in enumerate(urls):
        domain = urlparse(url).netloc.replace(".", "_")
        pdf_path = f"pdfs/{domain}_{i}.pdf"
        outline_path = f"outlines/{domain}_{i}_outline.json"

        # Try saving PDF
        success = await save_url_as_pdf(url, pdf_path)
        if not success:
            continue  # Skip to next URL

        outline = extract_outline_from_html(url)
        with open(outline_path, "w", encoding="utf8") as f:
            json.dump(outline, f, indent=2, ensure_ascii=False)

        features = extract_pdf_features(pdf_path)
        training_data = []

        for block in features:
            for page in block["text_blocks"]:
                training_data.append(page)
                training_data[-1]["level"] = "body"
                training_data[-1]["page"] = block["page_number"]

                if page["text"].strip() == re.sub(' +', ' ', outline["title"].strip()):
                    training_data[-1]["level"] = "title"
                    continue

                for answers in outline["outline"]:
                    if (
                        page["text"].strip() == re.sub(' +', ' ', answers["text"].strip())
                    ):
                        training_data[-1]["level"] = answers["level"]
                        break

        heading_levels = [d["level"] for d in training_data if d["level"].startswith("H")]
        if heading_levels:
            min_heading = min(int(h[1]) for h in heading_levels)
            for d in training_data:
                if d["level"].startswith("H"):
                    current_level = int(d["level"][1])
                    new_level = current_level - min_heading + 1
                    d["level"] = f"H{new_level}"

        prevdata += training_data
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(prevdata, f, indent=2, ensure_ascii=False)

    print(f"✅ Processed and saved to {json_path}")

# ---------------------- Main Entry ----------------------

if __name__ == "__main__":
    urls = [
    "https://www.un.org/en/about-us/un-charter/full-text",
    "https://www.nps.gov/aboutus/index.htm",
    "https://www.cdc.gov/nutrition/index.html",
    "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
    "https://www.who.int/health-topics/air-pollution#tab=tab_1",
    "https://www.unicef.org/what-we-do/education",
    "https://www.fao.org/hunger/en/",
    "https://www.worldbank.org/en/topic/poverty/overview",
    "https://www.epa.gov/climate-change-science",
    "https://www.imf.org/en/About",
    "https://www.nationalgeographic.com/science/article/history-of-earth",
    "https://www.si.edu/about",
    "https://www.loc.gov/about/",
    "https://www.britannica.com/topic/democracy",
    "https://www.history.com/topics/ancient-rome",
    "https://www.unwomen.org/en/what-we-do",
    "https://www.archives.gov/founding-docs/declaration-transcript",
    "https://kids.nationalgeographic.com/nature/article/ocean",
    "https://education.nationalgeographic.org/resource/ocean-currents/",
    "https://oceanservice.noaa.gov/education/kits/corals/coral01_intro.html",
    "https://www.usa.gov/voting",
    "https://www.state.gov/religious-freedom/",
    "https://www.usaid.gov/education",
    "https://www.pewresearch.org/global/2022/06/22/views-of-racial-inequality/",
    "https://www.census.gov/topics/population/age-and-sex.html",
    "https://www.un.org/en/global-issues/children",
    "https://www.un.org/en/global-issues/population",
    "https://www.un.org/en/about-us/universal-declaration-of-human-rights",
    "https://www.fao.org/nr/water/aquastat/water_use/index.stm",
    "https://www.worldwildlife.org/species",
    "https://www.space.com/24870-earth-planet.html",
    "https://climate.nasa.gov/causes/",
    "https://kids.nationalgeographic.com/nature/article/ocean",
    "https://education.nationalgeographic.org/resource/oceans/",
    "https://www.noaa.gov/education/resource-collections/climate/climate-change",
    "https://www.fbi.gov/investigate/civil-rights/hate-crimes",
    "https://oceanservice.noaa.gov/education/kits/corals/coral01_intro.html",
    "https://www.nps.gov/yose/planyourvisit/index.htm",
    "https://www.nps.gov/grca/learn/nature/index.htm",
    "https://www.si.edu/spotlight/native-american-women",
    "https://www.si.edu/spotlight/women-in-science",
    "https://www.khanacademy.org/humanities/world-history",
    "https://education.nationalgeographic.org/resource/ocean-currents/",
    "https://www.who.int/health-topics/malaria#tab=tab_1",
    "https://www.cdc.gov/disasters/hurricanes/index.html",
    "https://www.unodc.org/unodc/en/human-trafficking/what-is-human-trafficking.html",
    "https://www.cdc.gov/mentalhealth/learn/index.htm",
    "https://www.nih.gov/about-nih/what-we-do/mission-goals",
    "https://www.energy.gov/eere/buildings/articles/what-zero-energy-building",
    "https://www.noaa.gov/news/climate",
    "https://www.epa.gov/environmental-topics/land-and-soil",
    "https://www.usda.gov/topics/food-and-nutrition",
    "https://www.childwelfare.gov/topics/systemwide/statistics/",
    "https://www.federalreserve.gov/aboutthefed.htm",
    "https://www.pewresearch.org/religion/2021/09/28/religion-in-america/",
    "https://www.globalwitness.org/en/about-us/",
    "https://www.worldbank.org/en/news/all",
    "https://www.nps.gov/articles/ocean-tides.htm",
    "https://www.nps.gov/subjects/nationalregister/index.htm",
    "https://www.pbs.org/wgbh/nova/education/activities/3409_iceage.html",
    "https://www.pbs.org/wgbh/nova/education/resources/climate-change",
    "https://www.nationalparks.org/connect/blog/11-fun-facts-about-national-parks",
    "https://www.savethechildren.org/us/about-us/resource-library",
    "https://www.redcross.org/about-us/who-we-are.html",
    "https://www.amnesty.org/en/what-we-do/",
    "https://www.hud.gov/topics/rental_assistance/phprog",
    "https://www.unaids.org/en/what-we-do/",
    "https://www.unesco.org/en/education",
    "https://www.oas.org/en/about/who_we_are.asp",
    "https://www.childwelfare.gov/topics/parenting/",
    "https://www.un.org/en/global-issues/education",
    "https://www.fema.gov/press-release/20230425/fema-facts-building-future-less-risk",
    "https://www.nationalarchives.gov.uk/help-with-your-research/research-guides/",
    "https://www.developmentresearch.eu/?p=1139",
    "https://oceanservice.noaa.gov/education/kits/corals/coral01_intro.html",
    "https://education.nationalgeographic.org/resource/oceans/",
    "https://www.si.edu/spotlight/hispanic-heritage",
    "https://www.unca.edu/wrc/privacy/",
    "https://www.gnu.org/philosophy/free-sw.html",
    "https://plato.stanford.edu/entries/ethics-ai/",
    "https://www.kernel.org/doc/html/latest/",
    "https://www.fsf.org/about/what-is-free-software",
    "https://www.rfc-editor.org/rfc/rfc9110",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview",
    "https://www.gnu.org/software/bash/manual/bash.html",
    "https://sqlite.org/lang.html",
    "https://docs.python.org/3/tutorial/index.html",
    "https://man7.org/linux/man-pages/index.html",
    "https://www.w3.org/Provider/Style/URI",
    "https://xkcd.com/936/",
    "https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf",
    "https://mathworld.wolfram.com/",
    "https://www.cs.cmu.edu/~15131/f17/lectures/07-state/07-state.html",
    "https://csrc.nist.gov/glossary",
    "https://www.ietf.org/about/",
    "https://www.gnu.org/licenses/gpl-3.0.html",
    "https://www.oecd.org/about/",
    "https://learnxinyminutes.com/docs/python/",
    "https://www.nist.gov/itl/ssd/information-technology-laboratory",
    "https://www.cs.utexas.edu/users/EWD/ewd00xx/EWD8.PDF",
    "https://docs.djangoproject.com/en/5.0/topics/db/models/",
    "https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/syllabus/",
    "https://www.cs.princeton.edu/~rs/talks/LLRB/LLRB.pdf",
    "https://www.law.cornell.edu/constitution/first_amendment",
    "https://people.csail.mit.edu/rivest/Rsapaper.pdf",
    "https://gutenberg.org/files/1342/1342-h/1342-h.htm",
    "https://www.maths.cam.ac.uk/undergrad/admissions/worksheet-answers",
    "https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf",
    "https://www.irs.gov/pub/irs-pdf/p501.pdf",
    "https://www.iso.org/isoiec-27001-information-security.html",
    "https://docs.racket-lang.org/guide/index.html",
    "https://introcs.cs.princeton.edu/java/11style/",
    "https://cs3110.github.io/textbook/chapters/data/recursion.html",
    "https://www.msdmanuals.com/professional",
    "https://www.loc.gov/preservation/digital/formats/fdd/fdd000125.shtml",
    "https://www.cdc.gov/nchs/fastats/deaths.htm",
    "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2667623/",
    "https://www.ncbi.nlm.nih.gov/books/NBK430685/",
    "https://docs.kivy.org/en/stable/guide/lang.html",
    "https://wiki.archlinux.org/title/Systemd",
    "https://wiki.debian.org/Packaging",
    "https://tldp.org/LDP/abs/html/",
    "https://learn.microsoft.com/en-us/windows/win32/apiindex/windows-api-list",
    "https://docs.godotengine.org/en/stable/getting_started/step_by_step/index.html",
    "https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html",
    "https://opensource.com/resources/what-open-source",
    "https://help.libreoffice.org/latest/en-US/text/shared/guide/startcenter.html",
    "https://www.gnu.org/software/emacs/manual/html_node/emacs/index.html",
    "https://haskell.org/documentation/",
    "https://www.ocw.tudelft.nl/courses/",
    "https://www.khanacademy.org/computing/computer-science/algorithms",
    "https://www.coursera.org/learn/cryptography",
    "https://www.dataschool.io/python-pandas-tips-and-tricks/",
    "https://python-course.eu/numerical-programming/numpy-array-creation.php",
    "https://numpy.org/doc/stable/reference/index.html",
    "https://pytorch.org/docs/stable/tensors.html",
    "https://www.tensorflow.org/guide/tensor",
    "https://www.geeksforgeeks.org/data-structures/",
    "https://brilliant.org/wiki/number-theory/",
    "https://mathinsight.org/",
    "https://www.edx.org/course/data-science-machine-learning",
    "https://developers.google.com/machine-learning/glossary",
    "https://www.eff.org/issues/free-speech",
    "https://www.cs.columbia.edu/~hgs/audio/audio-intro.html",
    "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/",
    "https://cs.nyu.edu/courses/fall22/CSCI-UA.0101-001/lectures/lecture02.html",
    "https://stanfordnlp.github.io/CoreNLP/",
    "https://python-patterns.guide/",
    "https://www.openml.org/",
    "https://projecteuler.net/",
    "https://www.rosettacode.org/wiki/Rosetta_Code",
    "https://www.cs.ox.ac.uk/teaching/courses/2019-2020/csv/lectures/lecture2.pdf",
    "https://github.com/microsoft/ML-For-Beginners",
    "https://github.com/ossu/computer-science",
    "https://ai.googleblog.com/2024/05/introducing-gemma-open-models.html",
    "https://towardsdatascience.com/",
    "https://developers.cloudflare.com/",
    "https://learn.microsoft.com/en-us/dotnet/csharp/",
    "https://docs.oracle.com/en/java/",
    "https://tom.preston-werner.com/2011/11/22/readme-driven-development.html",
    "https://docs.github.com/en/pages",
    "https://opensource.guide/",
    "https://registry.npmjs.org/",
    "https://pkg.go.dev/std",
    "https://man.archlinux.org/",
    "https://devdocs.io/",
    "https://dev.to/",
    "https://www.linuxtopia.org/",
    "https://pub.towardsai.net/",
    "https://openai.com/research",
    "https://huggingface.co/docs/transformers/index",
    "https://www.cs.virginia.edu/~robins/GoToConsideredHarmful.html",
    "https://daringfireball.net/projects/markdown/syntax",
    "https://en.cppreference.com/w/",
    "https://llvm.org/docs/",
    "https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines",
    "https://www.boost.org/doc/libs/1_82_0/libs/optional/doc/html/index.html",
    "https://www.cplusplus.com/reference/",
    "https://www.kernel.org/doc/html/latest/filesystems/proc.html",
    "https://unix.stackexchange.com/questions/17495/how-do-i-use-the-proc-directory-to-see-information-about-my-system",
    "https://opensource.stackexchange.com/",
    "https://datascience.stackexchange.com/",
    "https://askubuntu.com/",
    "https://cs.stackexchange.com/",
    "https://superuser.com/questions/358024/what-is-the-difference-between-cpu-and-gpu",
    "https://meta.stackexchange.com/"
    "https://waitbutwhy.com/2014/05/fermi-paradox.html",
    "https://plato.stanford.edu/entries/skepticism/",
    "https://www.smashingmagazine.com/2020/01/web-typography-resources/",
    "https://www.gatesnotes.com/Books/How-to-Know-a-Person",
    "https://www.collaborativefund.com/blog/the-art-and-science-of-spending-money/",
    "https://fs.blog/mental-models/",
    "https://paulgraham.com/ds.html",
    "https://betterexplained.com/articles/understanding-recursion/",
    "https://distill.pub/2016/misread-tsne/",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Introduction",
    "https://ncase.me/neurons/",
    "https://ciechanow.ski/mechanical-watch/",
    "https://kottke.org/20/09/the-bizarre-true-story-of-the-boys-in-the-band",
    "https://every.to/napkin-math/airbnb-s-ipo-s-1-in-plain-english",
    "https://web.dev/articles/image-optimization",
    "https://www.nngroup.com/articles/f-shaped-pattern-reading-web-content/",
    "https://www.lesswrong.com/posts/rqQHgXt39RoYxxJkr/the-apprentice-s-path",
    "https://overreacted.io/how-does-react-tell-a-class-from-a-function/",
    "https://kottke.org/24/07/the-quest-to-build-americas-best-fall-color-app",
    "https://elephant.stockholm/2025/07/in-plain-text-reading-as-architecture/",
    "https://www.vox.com/features/2025/7/20/xyz-scholars-new-approach",
    "https://www.theatlantic.com/ideas/archive/2025/07/customer-service-sludge/683340/",
    "https://www.theatlantic.com/ideas/archive/2025/07/why-cant-americans-sleep/684010/",
    "https://www.theatlantic.com/ideas/archive/2025/06/secret-history-trump-private-cellphone/682160/",
    "https://www.theatlantic.com/ideas/archive/2025/06/worlds-hardest-bluffing-game/682120/",
    "https://www.theguardian.com/world/2025/jul/15/climate-crisis-oceans-tipping-points",
    "https://www.wired.com/story/ice-detention-center-911-emergencies/",
    "https://www.propublica.org/article/inside-the-mccain-sanders-clash",
    "https://www.wired.com/story/ai-art-philosophy-essay/",
    "https://www.newyorker.com/magazine/2025/06/30/your-hip-surgery-my-headache",
    "https://nautil.us/issue/107/the-future/is-human-superintelligence-possible",
    "https://aeon.co/essays/the-secret-intellectual-history-of-mathematics",
    "https://longreads.com/2025/05/15/the-bad-thing/",
    "https://longreads.com/2023/08/31/the-memory-picture/",
    "https://longreads.com/2016/10/05/the-family-that-would-not-live-2/",
    "https://aeon.co/essays/the-replica-and-the-original",
    "https://aeon.co/essays/memories-without-brains",
    "https://aeon.co/essays/the-power-of-the-c-word",
    "https://aeon.co/essays/the-french-liar",
    "https://aeon.co/essays/from-scattered-traces",
    "https://aeon.co/essays/the-future-homo-crustaceous",
    "https://aeon.co/essays/the-grammar-of-a-god-ocean",
    "https://aeon.co/essays/kind-of-confusing-is-consciousness-like-jazz",
    "https://aeon.co/essays/the-power-and-peril-of-altruism",
    "https://longreads.com/2019/12/20/longreads-best-of-2019-sports-writing/",
    "https://longreads.com/2018/07/26/leaving-a-good-man-is-hard-to-do/",
    "https://longreads.com/2021/12/16/best-of-2021-features/",
    "https://longreads.com/2022/12/20/best-of-2022-features/",
    "https://every.to/startup-wisdom/how-to-master-the-art-of-decisions",
    "https://www.discovermagazine.com/environment/how-lonely-earths-edges-are-causing-pandemics",
    "https://www.nature.com/articles/d41586-025-01724-5",
    "https://www.sciencedaily.com/releases/2025/07/25/major-climate-change-study.shtml",
    "https://medium.com/swlh/a-pattern-language-for-data-science-2025-6",
    "https://towardsdatascience.com/the-rise-of-composable-ai-in-2025-f43e8b9f1c2f",
    "https://blog.openai.com/gpt-4o-launch",
    "https://blog.google/technology/ai/next-snapshot-genai-2025-announcement",
    "https://blog.github.com/2025-07-20-advanced-code-search",
    "https://thenextweb.com/news/new-quantum-processor-breakthrough-2025",
    "https://venturebeat.com/ai/2025/07/23/ai-reinvention-customer-service",
    "https://techcrunch.com/2025/07/22/deep-learning-report-pub",
    "https://wired.com/2025/07/quantum-computing-encryption-deep-dive/",
    "https://nymag.com/intelligencer/2025/07/the-evolving-world-of-ai-memes.html",
    "https://slate.com/technology/2025/07/how-ai-is-changing-digital-art.html",
    "https://anthropocene.net/feature/an-oceanic-time-of-plastic/",
    "https://aeon.co/essays/a-good-conversation-relaxes-the-mind-and-opens-the-heart",
    "https://aeon.co/essays/why-the-great-books-still-speak-for-themselves-and-for-us",
    "https://aeon.co/essays/what-is-ethiopian-philosophy",
    "https://aeon.co/essays/a-cure-for-individualism",
    "https://www.cfr.org/report",
    "https://www.gmfus.org/publications",
    "https://www.wilsoncenter.org/publications",
    "https://www.cato.org/publications",
    "https://www.amnesty.org/en/latest/research/",
    "https://www.hrw.org/reports",
    "https://www.oxfam.org/en/research/",
    "https://www.doctorswithoutborders.org/latest/publications",
    "https://www.worldwildlife.org/publications",
    "https://www.greenpeace.org/international/reports/",
    "https://www.redcross.org/about-us/publications.html",
    "https://transparency.org/en/publications",
    "https://www.savethechildren.org/us/about-us/resource-library",
    "https://www.unhcr.org/publications",
    "https://www.mckinsey.com/insights/mckinsey-quarterly",
    "https://www.bcg.com/en-in/publications/default.aspx",
    "https://www.gartner.com/en/articles/latest",
    "https://www.forbes.com/innovation/",
    "https://hbr.org/tag/data-science",
    "https://www.zdnet.com/topic/artificial-intelligence/",
    "https://www.techcrunch.com/category/ai/",
    "https://www.wired.com/tag/ai/",
    "https://www.gartner.com/en/software/research",
    "https://www.statista.com/forecasts/worldwide",
    "https://www.loc.gov/collections/digital-collections/",
    "https://www.cia.gov/the-world-factbook/",
    "https://www.bl.uk/catalogues-and-collections/digital-collections",
    "https://www.sciencehistory.org/distillations/articles",
    "https://www.nps.gov/subjects/history/index.htm",
    "https://www.smithsonianmag.com/history/",
    "https://www.nationalgeographic.com/history/facts.html",
    "https://www.eia.gov/analysis/studies/",
    "https://aeon.co/essays/the-difference-between-rationality-and-intelligence",
    "https://longreads.com/2023/08/31/the-memory-picture/",
    "https://waitbutwhy.com/2014/05/fermi-paradox.html",
    "https://fs.blog/mental-models/",
    "https://betterexplained.com/articles/understanding-recursion/",
    "https://paulgraham.com/ds.html",
    "https://ncase.me/neurons/",
    "https://distill.pub/2016/misread-tsne/",
    "https://ciechanow.ski/mechanical-watch/",
    "https://www.lesswrong.com/posts/rqQHgXt39RoYxxJkr/the-apprentice-s-path",
    "https://overreacted.io/how-does-react-tell-a-class-from-a-function/",
    "https://web.dev/articles/image-optimization",
    "https://www.nngroup.com/articles/f-shaped-pattern-reading-web-content/",
    "https://www.collaborativefund.com/blog/the-art-and-science-of-spending-money/",
    "https://www.gatesnotes.com/Books/How-to-Know-a-Person",
    "https://every.to/napkin-math/airbnb-s-ipo-s-1-in-plain-english",
    "https://www.smashingmagazine.com/2020/01/web-typography-resources/",
    "https://www.lrb.co.uk/the-paper/v46/n14/thomas-laqueur/the-strange-death-of-elizabethan-england",
    "https://nautil.us/issue/109/frontiers/how-to-hack-your-beliefs",
    "https://www.propublica.org/article/boeing-max-737-faa-investigation",
    "https://restofworld.org/2023/chatgpt-education-kenya/",
    "https://www.ribbonfarm.com/2010/07/26/a-big-little-idea-called-legibility/",
    "https://meltingasphalt.com/interactive/going-critical/",
    "https://www.wired.com/story/silicon-valley-data-brokers/",
    "https://www.theatlantic.com/ideas/archive/2022/11/internet-life-online-privacy/672109/",
    "https://www.theguardian.com/news/2023/jul/13/the-tyranny-of-time-how-the-clock-rules-our-lives",
    "https://www.vox.com/the-highlight/24006622/clickbait-headlines-media-decline",
    "https://www.nytimes.com/2023/07/13/magazine/therapy-anxiety-depression.html",
    "https://www.economist.com/1843/2023/06/29/what-the-rise-of-ai-tells-us-about-intelligence",
    "https://www.bbc.com/future/article/20230626-the-surprising-downsides-of-being-clever",
    "https://www.brookings.edu/articles/the-automation-surge/",
    "https://stratechery.com/2023/the-end-of-the-tech-industry/",
    "https://www.technologyreview.com/2023/03/08/1069850/the-secret-police-inside-chinas-hackathon-scene/",
    "https://pluralistic.net/2023/06/18/the-broken-heart-of-domain-knowledge/",
    "https://marker.medium.com/the-ultimate-guide-to-launching-your-startup-on-product-hunt-52fd1e96b010",
    "https://www.edge.org/conversation/nicholas_a_christakis-how-evolution-helps-us-understand-human-social-networks",
    "https://future.a16z.com/ai-x-risk/",
    "https://www.quantamagazine.org/the-science-of-how-we-perceive-time-20230725/",
    "https://blog.codinghorror.com/the-best-definition-of-user-experience/",
    "https://arstechnica.com/features/2023/06/an-oral-history-of-the-mario-movie-that-almost-was/",
    "https://www.nature.com/articles/d41586-023-01465-6",
    "https://www.scientificamerican.com/article/the-ethical-dilemma-at-the-heart-of-eating-animals/",
    "https://www.3quarksdaily.com/3quarksdaily/2023/07/how-to-think-about-thinking.html",
    "https://www.ribbonfarm.com/2009/10/07/a-crash-course-in-structured-thinking/",
    "https://danluu.com/input-lag/",
    "https://rootsofprogress.org/why-did-we-wait-so-long-for-the-bicycle",
    "https://www.ethicalsystems.org/the-psychology-of-insider-trading/",
    "https://blog.samaltman.com/hard-startups",
    "https://jakobgreenfeld.com/persuasion-tactics",
    "https://sebastianraschka.com/blog/2023/llms-work.html",
    "https://www.developmentresearch.eu/?p=1139",
    "https://aeon.co/essays/is-it-time-to-chart-a-new-path-for-xenolinguistics-through-sci-fi",
    "https://aeon.co/essays/the-missing-women-of-autism-are-differently-different",
    "https://aeon.co/essays/our-crisis-is-not-loneliness-but-human-beings-becoming-invisible",
    "https://aeon.co/essays/why-one-branch-on-the-human-family-tree-replaced-all-the-others",
    "https://aeon.co/essays/commitment-and-cooperation-a-coevolutionary-relationship",
    "https://aeon.co/essays/how-it-became-wrong-for-nations-to-conquer-others",
    "https://aeon.co/essays/what-it-takes-to-be-a-glyph-breaker-deciphering-ancient-languages",
    "https://aeon.co/essays/why-identity-morality-and-faith-splinter-in-the-multiverse",
    "https://aeon.co/essays/the-replica-and-the-original",
    "https://aeon.co/essays/the-secret-intellectual-history-of-mathematics",
    "https://longreads.com/2025/07/21/puzzled/",
    "https://longreads.com/2025/07/10/becoming-earth/",
    "https://longreads.com/2025/03/21/going-soft/",
    "https://longreads.com/2025/02/27/solastalgia/",
    "https://longreads.com/2025/01/16/translators-notes/",
    "https://longreads.com/2025/02/18/after-lorne/",
    "https://longreads.com/2025/01/08/eastern-promises/",
    "https://www.theatlantic.com/ideas/archive/2025/07/silence-spiral/683372/",
    "https://www.theatlantic.com/ideas/archive/2025/07/trump-ukraine/683661/",
    "https://www.theatlantic.com/ideas/archive/2025/05/legalistic-noncompliance/682927/",
    "https://www.theatlantic.com/ideas/archive/2025/05/era-thrash/682919/",
    "https://www.theatlantic.com/ideas/archive/2025/07/trump-epstein/683544/",
    "https://www.theatlantic.com/ideas/archive/2025/02/woke-right/681716/",
    "https://www.theatlantic.com/ideas/archive/2025/01/amway-america/681479/",
    "https://www.theatlantic.com/ideas/archive/2025/02/trump-musk/681729/",
    "https://www.theatlantic.com/ideas/archive/2025/04/roganverse-split/682593/",
    "https://www.theatlantic.com/ideas/archive/2025/01/trump-sentenced/681271/",
    "https://aeon.co/essays/our-crisis-is-not-loneliness-but-human-beings-becoming-invisible",
    "https://aeon.co/essays/the-missing-women-of-autism-are-differently-different",
    "https://aeon.co/essays/why-one-branch-on-the-human-family-tree-replaced-all-the-others",
    "https://aeon.co/essays/commitment-and-cooperation-a-coevolutionary-relationship",
    "https://aeon.co/essays/how-it-became-wrong-for-nations-to-conquer-others",
    "https://aeon.co/essays/what-it-takes-to-be-a-glyph-breaker-deciphering-ancient-languages",
    "https://aeon.co/essays/why-identity-morality-and-faith-splinter-in-the-multiverse",
    "https://longreads.com/2025/07/21/puzzled/",
    "https://longreads.com/2025/07/10/becoming-earth/",
    "https://longreads.com/2025/03/21/going-soft/",
    "https://longreads.com/2025/02/27/solastalgia/",
    "https://longreads.com/2025/01/16/translators-notes/",
    "https://longreads.com/2025/02/18/after-lorne/",
    "https://www.theatlantic.com/ideas/archive/2025/07/silence-spiral/683372/",
    "https://www.theatlantic.com/ideas/archive/2025/07/trump-ukraine/683661/",
    "https://www.theatlantic.com/ideas/archive/2025/05/legalistic-noncompliance/682927/",
    "https://www.theatlantic.com/ideas/archive/2025/05/era-thrash/682919/",
    "https://www.theatlantic.com/ideas/archive/2025/07/trump-epstein/683544/",
    "https://www.theatlantic.com/ideas/archive/2025/02/woke-right/681716/",
    ]

    asyncio.run(run_pipeline(urls))
