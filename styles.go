package main

import "github.com/charmbracelet/lipgloss"

// Nord Polar Night (backgrounds)
const (
	nord0 = lipgloss.Color("#2E3440") // Darkest
	nord1 = lipgloss.Color("#3B4252")
	nord2 = lipgloss.Color("#434C5E")
	nord3 = lipgloss.Color("#4C566A") // Lightest bg
)

// Nord Snow Storm (foregrounds)
const (
	nord4 = lipgloss.Color("#D8DEE9")
	nord5 = lipgloss.Color("#E5E9F0")
	nord6 = lipgloss.Color("#ECEFF4") // Brightest fg
)

// Nord Frost (accents)
const (
	nord7  = lipgloss.Color("#8FBCBB") // Teal
	nord8  = lipgloss.Color("#88C0D0") // Light blue
	nord9  = lipgloss.Color("#81A1C1") // Medium blue
	nord10 = lipgloss.Color("#5E81AC") // Dark blue
)

// Nord Aurora (semantic)
const (
	nord11 = lipgloss.Color("#BF616A") // Red (errors)
	nord12 = lipgloss.Color("#D08770") // Orange (warnings)
	nord13 = lipgloss.Color("#EBCB8B") // Yellow (highlights)
	nord14 = lipgloss.Color("#A3BE8C") // Green (success)
	nord15 = lipgloss.Color("#B48EAD") // Purple (special)
)

type Styles struct {
	Header        lipgloss.Style // bg: nord1, fg: nord8, bold, full-width
	HeaderStatus  lipgloss.Style // fg: nord14 (idle), nord13 (working), nord11 (error)
	Viewport      lipgloss.Style // bg: nord0, fg: nord4, padding 1
	Footer        lipgloss.Style // bg: nord1, fg: nord5
	InputPrompt   lipgloss.Style // fg: nord8, bold ("❯ ")
	InputText     lipgloss.Style // fg: nord6
	ErrorText     lipgloss.Style // fg: nord11, italic
	CollectionTag lipgloss.Style // fg: nord15, bg: nord2, padding 0 1
	MainViewportFocused   lipgloss.Style // focused main viewport border
	MainViewportUnfocused lipgloss.Style // unfocused main viewport border
	RefViewportFocused    lipgloss.Style // focused references viewport border
	RefViewportUnfocused  lipgloss.Style // unfocused references viewport border
	SpinnerStyle  lipgloss.Style // fg: nord8
}

func DefaultStyles() Styles {
	return Styles{
		Header: lipgloss.NewStyle().
			Background(nord1).
			Foreground(nord8).
			Bold(true),
		HeaderStatus: lipgloss.NewStyle(),
		Viewport: lipgloss.NewStyle().
			Background(nord0).
			Foreground(nord4).
			Padding(1, 1),
		Footer: lipgloss.NewStyle().
			Background(nord1).
			Foreground(nord5),
		InputPrompt: lipgloss.NewStyle().
			Foreground(nord8).
			Bold(true),
		InputText: lipgloss.NewStyle().
			Foreground(nord6),
		ErrorText: lipgloss.NewStyle().
			Foreground(nord11).
			Italic(true),
		CollectionTag: lipgloss.NewStyle().
			Foreground(nord15).
			Background(nord2).
			Padding(0, 1),
		MainViewportFocused: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(nord8).
			Background(nord0).
			Foreground(nord4),
		MainViewportUnfocused: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(nord3).
			Background(nord0).
			Foreground(nord4),
		RefViewportFocused: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(nord8).
			Background(nord0).
			Foreground(nord4),
		RefViewportUnfocused: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(nord3).
			Background(nord0).
			Foreground(nord4),
		SpinnerStyle: lipgloss.NewStyle().
			Foreground(nord8),
	}
}

