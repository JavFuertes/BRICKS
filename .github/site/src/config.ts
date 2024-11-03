import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: "https://github.com/JavFuertes/BRICKS", // Replace this with your deployed domain if applicable
  author: "Javier Fuertes",
  profile: "https://github.com/JavFuertes/BRICKS",
  desc: "Tools for the assessment of masonry structures.",
  title: "BRICKS",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 3,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes, margin for scheduling posts
  showArchives: true, // Controls whether the Archives link appears in the navigation
};

export const LOCALE = {
  lang: "en", // HTML language code; default to "en" if not set
  langTag: ["en-EN"], // BCP 47 Language Tags; use environment default if empty
} as const;

export const LOGO_IMAGE = {
  enable: false, // Set to true if using a logo image
  svg: true, // Specify if the logo is in SVG format
  width: 216,
  height: 46,
};

export const SOCIALS: SocialObjects = [
  {
    name: "Github",
    href: "https://github.com/JavFuertes/BRICKS",
    linkTitle: `${SITE.title} on Github`,
    active: true,
  },
];
