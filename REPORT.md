#PhotoSymbols Scanner (PSS)

##Introduction

At Easy Read Online (ERO), we use either handcrafted illustrations or photo-based images in our documents. While illustrations are standard, we provide photo images when requested by clients. Over time, we built a large internal library of photo images sourced from Shutterstock and PhotoSymbols under an annual licence agreement.

##The Problemo

When renewing our PhotoSymbols licence this year, we discovered their updated terms no longer allowed competitors to renew subscriptions. This meant we could no longer use any PhotoSymbols images.

Although most standalone PhotoSymbols images were removed, many composite images had been created over the years using a mix of PhotoSymbols and Shutterstock elements. With over 5,000 total images and around 1,000 PhotoSymbols images to compare against, manually identifying affected composites was impractical, time-consuming, and unreliable — creating a risk of unintentionally using unlicensed content.

To solve this, I proposed adapting a Python tool I had previously developed (the “DIF”), which used CLIP embeddings to detect duplicate images. The plan was to modify it to compare our full image library against the PhotoSymbols set automatically.

##Why It Was Challenging

The solution was technically complex. Many images were composites, meaning elements could be resized, layered, cropped, or flattened, making direct matching difficult. File types also varied, adding another layer of complexity.

Additionally, the scale of the task — 5,000 images checked against 1,000 references — required an approach that was both accurate and computationally efficient.

