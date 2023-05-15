package org.vadere.state.attributes.models;

import java.util.Arrays;
import java.util.List;

import org.vadere.annotation.factories.attributes.ModelAttributeClass;
import org.vadere.state.attributes.Attributes;

@ModelAttributeClass
public class AttributesSIRG extends Attributes {

	private int infectionsAtStart = 10;  // set infectionsAtStart
	private double infectionRate = 0.01; // set infectionRate
	private double infectionMaxDistance = 1; // set infectionMaxDistance

	private double recoveryRate =0.1; // set recoveryRate

	public int getInfectionsAtStart() { return infectionsAtStart; }

	public double getInfectionRate() {
		return infectionRate;
	}

	public double getInfectionMaxDistance() {
		return infectionMaxDistance;
	}
	public double getRecoveryRate() { return  recoveryRate; } //get the value of recovery rate


}
