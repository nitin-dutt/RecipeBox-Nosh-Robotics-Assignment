import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom does not implement the native <dialog> methods the confirm dialog uses.
HTMLDialogElement.prototype.showModal ||= function (this: HTMLDialogElement) {
  this.setAttribute("open", "");
};
HTMLDialogElement.prototype.close ||= function (this: HTMLDialogElement) {
  this.removeAttribute("open");
};
Element.prototype.scrollIntoView ||= function () {};

afterEach(() => {
  cleanup();
  sessionStorage.clear();
});
